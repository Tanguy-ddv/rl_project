from session.ACsession import ACSession
import env
s = ACSession(env.SOURCE, 'outputs/compare', verbose=100)
s.load_last_agent(last_suffix='train_and_test/group_99')
s.test(10)