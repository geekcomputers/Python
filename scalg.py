
def score(source_data : list, weights : list, *args) -> list:

    '''
    int list - weights
    possible values - 0 / 1
    0 if lower values have higher weight in the data set
    1 if higher values have higher weight in the data set
    ==========
    Optional arguments:
    str - "score_lists"
    get a list with all the scores for each piece of data

    str - "scores"
    get only the final scores for each data set
    '''

    # getting data
    data_lists = []
    for item in source_data:
        for i in range(len(item)):
            try:
                data_lists[i].append(float(item[i]))
            except IndexError:
                data_lists.append([])
                data_lists[i].append(float(item[i]))

    score_lists = []
    # calculating price score
    for dlist, weight in zip(data_lists, weights):
        mind = min(dlist)
        maxd = max(dlist)

        score = []
        if weight == 0:
            for item in dlist:
                try:
                    score.append(1 - ((item - mind) / (maxd - mind)))
                except ZeroDivisionError:
                    score.append(1)

        elif weight == 1:
            for item in dlist:
                try:
                    score.append((item - mind) / (maxd - mind))
                except ZeroDivisionError:
                    score.append(0)

        else:
            raise ValueError("Invalid weight of %f provided" % (weight))

        score_lists.append(score)

    # return score lists
    if "score_lists" in args:
        return score_lists

    # initialize final scores
    final_scores = [0 for i in range(len(score_lists[0]))]

    # generate final scores
    for i, slist in enumerate(score_lists):
        for j, ele in enumerate(slist):
            final_scores[j] = final_scores[j] + ele

    # return only scores
    if "scores" in args:
        return final_scores

    # append scores to source data
    for i, ele in enumerate(final_scores):
        source_data[i].append(ele)

    return source_data
