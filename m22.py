from NewsSentiment import TargetSentimentClassifier
import json

def m22(input_fname, output_fname):
    f = open(input_fname)
    data = json.load(f)

    tsc = TargetSentimentClassifier()

    keywords = data['interestingTerms']

    def find_idx(text, word):
        idx = text.lower().find(word.lower())
        return idx, idx+len(word)

    user_body = data['match']['docs'][0]['body'][0]
    cluster_body_list = []

    num_articles = len(data['response']['docs'])

    for i in range(num_articles):
        cluster_body_list.append(data['response']['docs'][i]['body'][0])

    user_sentiments = {}
    for keyword in keywords:
        idxs = find_idx(user_body, keyword)
        try:
            user_sentiments[keyword] = tsc.infer(
                text=user_body,
                target_mention_from=idxs[0],
                target_mention_to=idxs[1]
            )[0]['class_label']
        except:
            pass
    user_sentiments

    keywords = list(user_sentiments.keys())
    
    keywords_indices = {}

    for keyword in keywords:
        keywords_indices[keyword] = [find_idx(cluster_body_list[i], keyword) for i in range(num_articles)]

    topic_sentiments = {}
    
    for keyword in keywords_indices:
        topic_sentiments[keyword] = []
        for i in range(num_articles):
            try:
                topic_sentiments[keyword].append(tsc.infer(
                    text=cluster_body_list[i],
                    target_mention_from=keywords_indices[keyword][i][0],
                    target_mention_to=keywords_indices[keyword][i][1]
                )[0]['class_label'])
            except:
                topic_sentiments[keyword].append('')
    
    output = {}

    keywords_list = []
    for keyword in keywords:
        keywords_list.append({'word': keyword,
                            'sentiment': user_sentiments[keyword]})

    output['user_article'] = {
        'title': data['match']['docs'][0]['headline'][0],
        'source': data['match']['docs'][0]['outlet'][0],
        'politicalLeaning': data['match']['docs'][0]['political_leaning'][0],
        'keywords': keywords_list
    }

    articles_list = {}
    for article in range(num_articles):
        keywords_list = []
        for keyword in keywords:
            keywords_list.append({'word': keyword,
                                'sentiment': topic_sentiments[keyword][article]})
        articles_list[article] = {
            'title': data['response']['docs'][article]['headline'][0],
            'source': data['response']['docs'][article]['outlet'][0],
            'politicalLeaning': data['response']['docs'][article]['political_leaning'][0],
            'keywords': keywords_list
        }

    output['queried_articles'] = articles_list

    with open(output_fname, 'w') as f:
        json.dump(output, f)