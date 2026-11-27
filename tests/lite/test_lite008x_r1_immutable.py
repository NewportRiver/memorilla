import copy
import json
import tempfile
import unittest
from pathlib import Path
from experiments.lite008x_r1_immutable import (IntegrityError, canonical, digest,
    evaluate, file_hash, verify_bundle)


class ImmutableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundles = []
        for name in ('encoder', 'decoder'):
            root = self.root/name; root.mkdir()
            for f in ('weights.bin', 'config.json', 'tokenizer.json'):
                (root/f).write_bytes((name+f).encode())
            self.bundles.append({'root': str(root), 'files': {p.name: file_hash(p) for p in root.iterdir()}, 'settings': {'cpu': True, 'threads': 1}})
        enc, dec = map(verify_bundle, self.bundles)
        self.identity = dict(manifest_sha256='a'*64, partition='development', entity='e', query='q', evidence='e=A', candidate_answers=['A','B'], candidate_token_ids=[1,2], factual_direction='A', encoder_identity=enc, decoder_identity=dec, generation_config={'source':'fixture'})
        self.obj = dict(dtype='float32',shape=[2,2],W=[[1,0],[0,2]],eos=[1,1],encoder_identity=enc,decoder_identity=dec)
        self.path = self.root/'projection.json'; self.save()

    def save(self):
        self.path.write_text(json.dumps(self.obj)); self.sha = file_hash(self.path)

    def run_eval(self, **kwargs):
        args = dict(path=self.path,expected_sha256=self.sha,shape=(2,2),encoder=self.bundles[0],decoder=self.bundles[1],identity=self.identity,feature=[2,3],expected_input_sha256=digest(canonical({'identity':self.identity,'feature':[2,3]})))
        args.update(kwargs); return evaluate(**args)

    def test_valid_and_repeat(self):
        a=self.run_eval(); self.assertEqual(a,self.run_eval()); self.assertEqual(a['output'],[3,7]); self.assertEqual(file_hash(self.path),self.sha)
    def test_bad_sha(self):
        with self.assertRaises(IntegrityError): self.run_eval(expected_sha256='0'*64)
    def test_missing_sha(self):
        with self.assertRaises(IntegrityError): self.run_eval(expected_sha256=None)
    def test_dtype(self):
        self.obj['dtype']='float64'; self.save()
        with self.assertRaises(IntegrityError): self.run_eval()
    def test_shape(self):
        self.obj['W'][0].append(0); self.save()
        with self.assertRaises(IntegrityError): self.run_eval()
    def test_nonfinite(self):
        for value in (float('nan'),float('inf'),1e100):
            self.obj['W'][0][0]=value; self.save()
            with self.assertRaises(IntegrityError): self.run_eval()
    def test_input_hash(self):
        with self.assertRaises(IntegrityError): self.run_eval(expected_input_sha256='0'*64)
    def test_missing_identity(self):
        del self.identity['entity']
        with self.assertRaises(IntegrityError): self.run_eval()
    def test_model_identity(self):
        self.obj['encoder_identity']='wrong'; self.save()
        with self.assertRaises(IntegrityError): self.run_eval()
    def test_changed_decoder_weight(self):
        (Path(self.bundles[1]['root'])/'weights.bin').write_bytes(b'changed')
        with self.assertRaises(IntegrityError): self.run_eval()
    def test_extra_model_file(self):
        (Path(self.bundles[0]['root'])/'shard.bin').write_bytes(b'omitted')
        with self.assertRaises(IntegrityError): self.run_eval()
    def test_projection_post_mutation(self):
        with self.assertRaises(IntegrityError): self.run_eval(_after_load=lambda:self.path.write_bytes(b'changed'))
    def test_model_post_mutation(self):
        with self.assertRaises(IntegrityError): self.run_eval(_after_load=lambda:(Path(self.bundles[0]['root'])/'weights.bin').write_bytes(b'changed'))
    def test_same_case_different_content(self):
        original=digest(canonical({'identity':self.identity,'feature':[2,3]})); self.identity['query']='different query'
        with self.assertRaises(IntegrityError): self.run_eval(expected_input_sha256=original)
    def test_symlink(self):
        (Path(self.bundles[0]['root'])/'link').symlink_to('weights.bin')
        with self.assertRaises(IntegrityError): self.run_eval()


if __name__ == '__main__': unittest.main()
