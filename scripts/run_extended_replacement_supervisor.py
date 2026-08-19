"""Detached sequential supervisor for the replacement extended workflow."""
from __future__ import annotations
import json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT/"outputs/static_phenotype_trajectory_stratified_v2/extended_clinical_latent_dynamics_v1"
LOG=EXT/"logs"; ARCH=EXT/"timestamped_state_archive_35072d"

def now(): return datetime.now(timezone.utc).isoformat()
def write(x): (LOG/"replacement_supervisor_progress.json").write_text(json.dumps(x,indent=2)+"\n")
def exporter_running():
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:
   if b'export_timestamped_states_35072d.py' in p.read_bytes(): return True
  except (FileNotFoundError,PermissionError,ProcessLookupError): pass
 return False

def main():
 LOG.mkdir(parents=True,exist_ok=True); marker=EXT/"STATE_ARCHIVE_COMPLETE.json"
 while not marker.exists():
  if not exporter_running():
   write({"updated_at":now(),"status":"blocked","stage":"state_export","reason":"Exporter exited without completion marker"}); return 2
  write({"updated_at":now(),"status":"waiting","stage":"state_export"}); time.sleep(30)
 validation=json.loads((ARCH/"canonical_representation_validation.json").read_text())
 if not validation.get('passed'):
  write({"updated_at":now(),"status":"blocked","stage":"representation_validation","validation":validation}); return 3
 phases=[
  ("phases_2_and_3","run_extended_circadian_dynamics.py"),
  ("phase_4","run_extended_event_rewiring.py"),
  ("final_qa","finalize_extended_clinical_latent_analysis.py"),
 ]
 for stage,script in phases:
  write({"updated_at":now(),"status":"running","stage":stage,"script":script})
  with (LOG/f"{stage}.log").open('a') as handle:
   result=subprocess.run([str(Path('/home/myriamcharfeddine/miniconda3/envs/ssmcgm/bin/python')),str(ROOT/'scripts'/script)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
  if result.returncode:
   write({"updated_at":now(),"status":"failed","stage":stage,"script":script,"returncode":result.returncode}); return result.returncode
 write({"updated_at":now(),"status":"complete","stage":"all","final_qa":str(EXT/'FINAL_QA_REPORT.md')}); return 0

if __name__=='__main__': raise SystemExit(main())
