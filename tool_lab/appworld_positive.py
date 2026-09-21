"""Publicly grounded successful alternatives for the AppWorld teaching slice."""
import copy


def proposals(reference, point):
    trace = reference['trace']
    if point not in reference['points']:
        raise ValueError('Undeclared history')
    result = {}
    observations = [e for e in trace[:point] if e['method']=='get' and e['app']!='supervisor']
    if observations:
        result['repeat_observation'] = trace[:point] + [copy.deepcopy(observations[-1])] + trace[point:]
    if 'access_token' in trace[point]['arguments']:
        failed = copy.deepcopy(trace[point])
        failed['arguments']['access_token'] = 'invalid-local-control'
        result['recover_authentication'] = trace[:point] + [failed] + trace[point:]
    return result
