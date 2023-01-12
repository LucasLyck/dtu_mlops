import pstats

p = pstats.Stats('profile.log')
p.strip_dirs().sort_stats('time').print_stats()
