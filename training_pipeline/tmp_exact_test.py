def f(out, comps):
    exact = bool(comps.get('exact_set_match'))
    out['success'] = exact
    return out
