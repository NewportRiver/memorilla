import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pytest
from lite.memory.provenance import SourceRecord,EncoderContract

def test_source_identity_and_idempotency_key_change_on_contract_or_content():
 s=SourceRecord.from_content('e1','same',provenance='fixture',source_revision='1',access_scope='public',source_timestamp='t',collection_id='c')
 c=EncoderContract('mini','r1','a'*64,'tok','prep'); assert s.verify(); k=c.key(s); assert k==c.key(s)
 s2=SourceRecord.from_content('e1','changed',provenance='fixture',source_revision='2',access_scope='public',source_timestamp='t',collection_id='c'); assert k!=c.key(s2)
 c2=EncoderContract('mini','r2','b'*64,'tok','prep'); assert k!=c2.key(s)

def test_tampered_source_rejected():
 s=SourceRecord('e1','tampered','0'*64,'fixture','1','public','t','c')
 with pytest.raises(ValueError,match='source hash mismatch'): s.verify()

def test_invalid_identity_rejected():
 with pytest.raises(ValueError): SourceRecord.from_content('','x',provenance='p',source_revision='1',access_scope='public',source_timestamp='t',collection_id='c').verify()
