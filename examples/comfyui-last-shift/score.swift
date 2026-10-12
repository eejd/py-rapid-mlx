// SPDX-License-Identifier: Apache-2.0
// Original sparse piano cue, rendered offline through macOS's bundled instrument.
import AVFoundation
import Foundation

struct Note { let time: Double; let pitch: UInt8; let velocity: UInt8; let length: Double }
let notes: [Note] = [
    Note(time:0.35,pitch:50,velocity:39,length:4),
    Note(time:1.2,pitch:69,velocity:42,length:2.5),
    Note(time:3.1,pitch:65,velocity:38,length:2.7),
    Note(time:5.4,pitch:64,velocity:35,length:2.7),
    Note(time:7.5,pitch:57,velocity:35,length:3),
    Note(time:8.3,pitch:62,velocity:42,length:3),
    Note(time:10.8,pitch:50,velocity:39,length:3.7),
    Note(time:11.2,pitch:65,velocity:44,length:3),
    Note(time:12.7,pitch:69,velocity:41,length:2.8),
    Note(time:14.1,pitch:72,velocity:46,length:3.2),
    Note(time:16.4,pitch:64,velocity:32,length:2.2),
    // Let the shutdown breathe; the same motif returns in F major.
    Note(time:20.0,pitch:53,velocity:43,length:4.5),
    Note(time:20.65,pitch:65,velocity:44,length:3.5),
    Note(time:21.55,pitch:69,velocity:48,length:3),
    Note(time:22.45,pitch:72,velocity:46,length:3),
    Note(time:23.5,pitch:74,velocity:40,length:3),
    Note(time:24.4,pitch:60,velocity:37,length:4.5),
    Note(time:24.6,pitch:69,velocity:39,length:4.2),
    Note(time:25.6,pitch:77,velocity:40,length:3.8)
]
let output = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "score.wav"
let engine = AVAudioEngine()
let sampler = AVAudioUnitSampler()
let reverb = AVAudioUnitReverb()
reverb.loadFactoryPreset(.mediumHall)
reverb.wetDryMix = 19
engine.attach(sampler); engine.attach(reverb)
let format = AVAudioFormat(standardFormatWithSampleRate:48000,channels:2)!
engine.connect(sampler,to:reverb,format:format)
engine.connect(reverb,to:engine.mainMixerNode,format:format)
try sampler.loadSoundBankInstrument(at:URL(fileURLWithPath:"/System/Library/Components/CoreAudio.component/Contents/Resources/gs_instruments.dls"),program:0,bankMSB:121,bankLSB:0)
try engine.enableManualRenderingMode(.offline,format:format,maximumFrameCount:1024)
try engine.start()
let file = try AVAudioFile(forWriting:URL(fileURLWithPath:output),settings:format.settings)
let buffer = AVAudioPCMBuffer(pcmFormat:engine.manualRenderingFormat,frameCapacity:1024)!
struct Event {let frame:Int64;let pitch:UInt8;let velocity:UInt8}
var events:[Event] = []
for n in notes {
    events.append(Event(frame:Int64(n.time*48000),pitch:n.pitch,velocity:n.velocity))
    events.append(Event(frame:Int64((n.time+n.length)*48000),pitch:n.pitch,velocity:0))
}
events.sort{$0.frame < $1.frame}
var index=0
let end:Int64=Int64(31*48000)
while engine.manualRenderingSampleTime < end {
    let now = engine.manualRenderingSampleTime
    while index < events.count && events[index].frame <= now {
        let e=events[index]
        if e.velocity > 0 {sampler.startNote(e.pitch,withVelocity:e.velocity,onChannel:0)}
        else {sampler.stopNote(e.pitch,onChannel:0)}
        index += 1
    }
    let next = index < events.count ? min(events[index].frame,end) : end
    let count = AVAudioFrameCount(min(1024,next-now))
    if count == 0 {continue}
    let status=try engine.renderOffline(count,to:buffer)
    switch status {
    case .success: try file.write(from:buffer)
    case .cannotDoInCurrentContext: continue
    default: fatalError("Offline render failed: \(status)")
    }
}
engine.stop()
print(output)
