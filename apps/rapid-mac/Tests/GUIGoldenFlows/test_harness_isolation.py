"""Run with python3 -m unittest discover -s Tests/GUIGoldenFlows -p 'test_*.py'."""
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / 'scripts/gui-golden-flows.sh').read_text()


def function(name):
    return re.search(r'^' + name + r'\(\) \{\n.*?^\}', SCRIPT, re.M | re.S)[0]


class HarnessIsolationTests(unittest.TestCase):
    def test_feature_inventory_matches_product_defaults(self):
        table = subprocess.check_output(
            ['bash', '-c', function('persona_feature_defaults') + '\npersona_feature_defaults'],
            text=True,
        )
        seeds = dict((key, (kind, value)) for key, kind, value in
                     (line.split() for line in table.splitlines()))
        configs = list((ROOT / 'Sources/Rapid').rglob('*FeatureConfig.swift'))
        self.assertGreaterEqual(len(configs), 5)
        for config in configs:
            text = config.read_text()
            key = re.search(r'static let enabledKey = "([^"]+)"', text)[1]
            default = re.search(r'static let defaultEnabled = (true|false)', text)[1]
            self.assertEqual(seeds[key], ('bool', default), str(config))
        # MTP uses a per-model JSON map, not a global on/off flag. An empty
        # map preserves the product/catalog default and allows UI opt-in.
        perf = (ROOT / 'Sources/Rapid/Server/ModelPerfConfigStore.swift').read_text()
        perf_key = re.search(r'static let storageKey = "([^"]+)"', perf)[1]
        self.assertEqual(seeds[perf_key][0], 'data')
        self.assertEqual(json.loads(bytes.fromhex(seeds[perf_key][1])), {})
        for name in ('web_search', 'browse', 'weather', 'read_document',
                     'local_search', 'local_read', 'local_write', 'local_trash', 'local_run'):
            self.assertEqual(seeds[f'rapid.tools.enabled.{name}'], ('bool', 'true'))

    def run_wait(self, sequence):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for index, state in enumerate(sequence):
                elements = [] if state == 'missing' else [
                    {'identifier': identifier, **({} if state == 'unreadable' else
                                                 {'enabled': state == 'ready'})}
                    for identifier in ('Sidebar.Residency.Unload', 'ChatView.Residency.Unload')
                ]
                (directory / f'{index}.json').write_text(json.dumps({'data': {'ui_elements': elements}}))
            shell = function('wait_residency_idle') + '''
index=0
see_main() {
    local fixture=$index
    [[ "$fixture" -lt "$limit" ]] || fixture=$((limit - 1))
    cp "$directory/$fixture.json" "$1"
    index=$((index + 1))
}
sleep() { :; }
die() { echo "$*" >&2; exit 1; }
wait_residency_idle "$directory/result.json"
echo "$index"
'''
            return subprocess.run(['bash', '-eu', '-c', shell], text=True, capture_output=True,
                                  env={'PATH': '/opt/homebrew/bin:/usr/bin:/bin',
                                       'directory': str(directory), 'limit': str(len(sequence))})

    def test_residency_wait_resets_after_busy_or_missing_reads(self):
        result = self.run_wait(['ready', 'ready', 'busy', 'ready', 'missing',
                                'ready', 'unreadable', 'ready', 'ready', 'ready', 'ready'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '11')

    def test_stably_disabled_or_unreadable_controls_never_pass(self):
        for state in ('busy', 'missing', 'unreadable'):
            result = self.run_wait([state])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('did not settle enabled', result.stderr)


if __name__ == '__main__':
    unittest.main()
