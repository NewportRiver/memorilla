from lite.training.orchestration import EpisodeBudget

def test_interrupted_episode_consumes_uncertain_authorization(tmp_path):
    p=tmp_path/'budget.json'; b=EpisodeBudget(p,8,'d','c'); b.begin_episode('e',8); b.record_update('e'); b.finish_episode('e','interrupted')
    state=EpisodeBudget(p,8,'d','c'); assert state.state['used_updates']==8; assert state.remaining_updates==0; assert state.state['episodes'][0]['last_phase']=='interrupted_conservatively_consumed'

def test_interrupted_episode_never_increases_budget(tmp_path):
    p=tmp_path/'budget.json'; b=EpisodeBudget(p,8,'d','c'); b.begin_episode('e',8); before=b.remaining_updates; b.finish_episode('e','interrupted'); assert b.remaining_updates < before
