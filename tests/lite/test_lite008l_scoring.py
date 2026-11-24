from experiments.lite008l.scoring import answer_margin,answer_positions,counterfactual_success

def test_answer_margin_orientation():
 assert answer_margin(-2.0,-3.0)==1.0
 assert counterfactual_success(-0.1)
 assert not counterfactual_success(0.1)

def test_answer_positions_cover_full_answer():
 assert answer_positions(5,1)==(4,4)
 assert answer_positions(5,2)==(4,5)

def test_invalid_empty_sequences_rejected():
 for args in [(0,1),(1,0)]:
  try: answer_positions(*args)
  except ValueError: pass
  else: raise AssertionError(args)
