import CryptoKit
import Darwin
import DeviceCheck
import Foundation
import Security

// Native cryptographic endpoint only; there is intentionally no model or
// forwarding to a Python/plaintext HTTP sidecar in this experiment.
private let domain = "rapid-provider-attestation-experiment-v1"

private struct Refused: Error { let code: String }

private func emit(_ value: [String: Any]) {
  let data = try! JSONSerialization.data(
    withJSONObject: value, options: [.sortedKeys, .withoutEscapingSlashes])
  FileHandle.standardOutput.write(data)
  FileHandle.standardOutput.write(Data([10]))
}

private func bytes(_ value: Any?, count: Int? = nil) throws -> Data {
  guard let value = value as? String, let data = Data(base64Encoded: value),
    count == nil || data.count == count
  else { throw Refused(code: "invalid_encoding") }
  return data
}

private func codeIdentity() -> [String: Any] {
  var code: SecCode?
  var staticCode: SecStaticCode?
  var info: CFDictionary?
  guard SecCodeCopySelf([], &code) == errSecSuccess, let code,
    SecCodeCopyStaticCode(code, [], &staticCode) == errSecSuccess, let staticCode,
    SecCodeCopySigningInformation(staticCode, SecCSFlags(rawValue: kSecCSSigningInformation), &info)
      == errSecSuccess,
    let values = info as? [String: Any]
  else {
    return ["signed": false, "hardened_runtime": false, "official_identity": false]
  }
  let flags = (values[kSecCodeInfoFlags as String] as? NSNumber)?.uint32Value ?? 0
  let team = values[kSecCodeInfoTeamIdentifier as String] as? String
  return [
    "signed": true, "ad_hoc": flags & 0x2 != 0,
    "hardened_runtime": flags & 0x10000 != 0,
    "has_team_identity": team != nil,
    // An entitlement/readable Team ID is not a server-verified proof.
    "official_identity": false,
  ]
}

private func enclaveProbe() -> [String: Any] {
  guard SecureEnclave.isAvailable else { return ["available": false, "sign_verify": false] }
  do {
    // Ephemeral handle: no permanent named Keychain item is created.
    let key = try SecureEnclave.P256.Signing.PrivateKey()
    let message = Data("rapid-attestation-feasibility-probe".utf8)
    let signature = try key.signature(for: message)
    return [
      "available": true, "sign_verify": key.publicKey.isValidSignature(signature, for: message),
      "apple_attestation": false,
    ]
  } catch {
    return ["available": true, "sign_verify": false, "error": "enclave_operation_failed"]
  }
}

@main
struct ProviderProbe {
  static func main() async {
    var limit = rlimit(rlim_cur: 0, rlim_max: 0)
    let coreDisabled = setrlimit(RLIMIT_CORE, &limit) == 0
    let key = Curve25519.KeyAgreement.PrivateKey()
    let supported = DCAppAttestService.shared.isSupported
    let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
    guard let executableData = try? Data(contentsOf: executable) else {
      emit(["error": "cannot_measure_executable"])
      exit(1)
    }
    let digest = SHA256.hash(data: executableData).map { String(format: "%02x", $0) }.joined()
    emit([
      "type": "hello", "protocol": domain,
      "public_key": key.publicKey.rawRepresentation.base64EncodedString(),
      "build": digest, "app_attest_supported": supported,
      "os": ProcessInfo.processInfo.operatingSystemVersionString,
      "code_identity": codeIdentity(), "core_dumps_disabled": coreDisabled,
      "secure_enclave": enclaveProbe(), "apple_verified": false,
      "inference": false,
    ])
    if CommandLine.arguments.contains("--probe") { return }

    var seen = Set<String>()
    while let line = readLine() {
      do {
        guard line.utf8.count <= 65536,
          let request = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
          let operation = request["operation"] as? String
        else { throw Refused(code: "invalid_command") }
        if operation == "attest" {
          // This is a native capability probe, NOT server validation.
          guard supported else { throw Refused(code: "app_attest_unsupported") }
          guard let transcript = request["transcript"] as? [String: Any],
            transcript["public_key"] as? String
              == key.publicKey.rawRepresentation.base64EncodedString(),
            transcript["domain"] as? String == domain,
            transcript["proof_kind"] as? String == "apple",
            let nonce = transcript["nonce"] as? String, nonce.utf8.count == 64
          else {
            throw Refused(code: "invalid_attestation_binding")
          }
          let encoded = try JSONSerialization.data(
            withJSONObject: transcript, options: [.sortedKeys, .withoutEscapingSlashes])
          let hash = Data(SHA256.hash(data: encoded))
          let keyID = try await DCAppAttestService.shared.generateKey()
          let proof = try await DCAppAttestService.shared.attestKey(keyID, clientDataHash: hash)
          emit([
            "type": "apple_evidence", "key_id": keyID,
            "proof": proof.base64EncodedString(), "server_verified": false,
          ])
          continue
        }
        guard operation == "open",
          let envelope = request["envelope"] as? [String: Any],
          let binding = envelope["binding"] as? [String: Any],
          binding["domain"] as? String == domain,
          binding["proof_kind"] as? String == "fixture-only",
          binding["public_key"] as? String == key.publicKey.rawRepresentation.base64EncodedString(),
          binding["build"] as? String == digest,
          let requestID = binding["request_id"] as? String
        else { throw Refused(code: "endpoint_mismatch") }
        guard !seen.contains(requestID) else { throw Refused(code: "request_replayed") }
        // Bounded experiment, fail closed instead of evicting replay IDs.
        guard seen.count < 256 else { throw Refused(code: "session_capacity") }
        let context = try JSONSerialization.data(
          withJSONObject: binding, options: [.sortedKeys, .withoutEscapingSlashes])
        let sender = try Curve25519.KeyAgreement.PublicKey(
          rawRepresentation: bytes(envelope["sender_key"], count: 32))
        let shared = try key.sharedSecretFromKeyAgreement(with: sender)
        let symmetric = shared.hkdfDerivedSymmetricKey(
          using: SHA256.self,
          salt: Data(SHA256.hash(data: context)), sharedInfo: Data(domain.utf8), outputByteCount: 32
        )
        let combined = try bytes(envelope["ciphertext"])
        guard combined.count >= 16, combined.count <= 16400 else {
          throw Refused(code: "payload_too_large")
        }
        let nonce = try AES.GCM.Nonce(data: bytes(envelope["nonce"], count: 12))
        let box = try AES.GCM.SealedBox(
          nonce: nonce, ciphertext: combined.dropLast(16), tag: combined.suffix(16))
        let plaintext = try AES.GCM.open(
          box, using: symmetric, authenticating: context + Data("|request".utf8))
        seen.insert(requestID)
        // Synthetic transform proves the native process decrypted it;
        // no model output or customer content is persisted or emitted.
        let result: [String: Any] = [
          "bytes": plaintext.count, "inference": false,
          "sha256": SHA256.hash(data: plaintext).map { String(format: "%02x", $0) }.joined(),
        ]
        let encoded = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
        let reply = try AES.GCM.seal(
          encoded, using: symmetric, authenticating: context + Data("|response".utf8))
        emit([
          "type": "encrypted_response",
          "nonce": reply.nonce.withUnsafeBytes { Data($0) }.base64EncodedString(),
          "ciphertext": (reply.ciphertext + reply.tag).base64EncodedString(),
        ])
      } catch let error as Refused {
        emit(["error": error.code, "apple_verified": false])
      } catch {
        // Never reflect framework exceptions, prompt bytes or credentials.
        emit(["error": "operation_failed", "apple_verified": false])
      }
    }
  }
}
