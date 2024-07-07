def rolling_avg(values: list[float], n: int):
    return [sum(values[n*i: n*(i+1)])/n for i in range(len(values)//n)]