def ModelIt(fromUser  = 'Default', population = 0):
  print 'The population is %i' % population
  result = float(population)/1000000.0
  if fromUser != 'Default':
    return result
  else:
    return 'check your input'