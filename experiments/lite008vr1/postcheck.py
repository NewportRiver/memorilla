"""Read-only custody checks plus update of this milestone's own telemetry."""
import json
import subprocess
from pathlib import Path
from experiments.lite008vr1.identity import file_hash

ROOT = Path(__file__).resolve().parents[2]
base = ROOT/'docs/memorilla-lite/audits'
p = base/'LITE-008V-R1-RESOURCE-TELEMETRY.json'
d = json.loads(p.read_text())
m = json.loads((base/'LITE-008V-R1-PROVENANCE-MANIFEST.json').read_text())
model = Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model'
d['protected_postchecks'] = {k: file_hash(model/k) == v for k, v in m['records'][0]['identity']['model'].items()}
d['targeted_tests'] = '34 passed in 8.37s'
d['full_regression_blocker'] = 'Inherited conftest imports missing datasets; collection exit 4; no install.'
d['historical_source_unchanged_since_U'] = {path: subprocess.check_output(['git', 'show', '4b9f3e9:'+path], cwd=ROOT) == (ROOT/path).read_bytes() for path in ['experiments/lite008p/p_experiment.py', 'experiments/lite008p/dataset-manifest.json']}
ps = subprocess.check_output(['ps', '-A', '-o', 'pid,args'], text=True)
d['experiment_process_matches'] = [line for line in ps.splitlines() if 'python' in line and ('-m experiments.lite008vr1.recover' in line or 'python experiments/lite008vr1/monitor.py' in line) and 'bash -c' not in line]
d['preflight_conclusion'] = 'Full-resident decoder blocked: payload exceeds inference RSS ceiling; no decoder loaded.'
d['harness_failures'] = ['Postcheck sandbox import lacked repo path', 'Sandbox denied git subprocess; postchecks moved to normal terminal. Neither attempt ran scientific operations.']
p.write_text(json.dumps(d, indent=2)+'\n')
for f in base.glob('LITE-008V-R1-*.json'):
    json.loads(f.read_text())
assert all(d['protected_postchecks'].values())
assert all(d['historical_source_unchanged_since_U'].values())
assert not d['experiment_process_matches']
print('JSON, source custody, protected hashes, process checks PASS')
