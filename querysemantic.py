from __future__ import print_function
import sys
import json
import preprocess as p
import urllib

def query_keyword(keyword):
    api_key = ""#open('.api_key').read()
    service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
    params = {
        'query': keyword,
        'limit': 3,
        'indent': True,
        'key': api_key,
    }
    url = service_url + '?' + urllib.parse.urlencode(params)
    response = json.loads(urllib.request.urlopen(url).read())
    # print('response: ',response)
    all_results=[]
    about_keyword=[]
    for element in response['itemListElement']:
        about_keyword.append(element['result']['name'])
        # result_name=element['result']['name']
        # result_description=''
        # detailed_description=''
        if 'description' in element['result']:
            about_keyword.append(element['result']['description'])
            # result_description=element['result']['description']
        if 'detailedDescription' in element['result']:
            # print(p.preprocess(element['result']['detailedDescription']['articleBody']))
            about_keyword.append(' '.join(p.preprocess(element['result']['detailedDescription']['articleBody'])))
            # detailed_description=element['result']['detailedDescription']['articleBody']
        # print(' '.join(about_keyword))
        # test=result_name+' '+result_description+' '+detailed_description
        # all_results.append(' '.join(test))
        # all_results.append(test)
    # return all_results
    return about_keyword

def main(keyword):
    return (' '.join(query_keyword(keyword)))
    # return query_keyword(keyword)
if __name__ =='__main__':
    keyword=sys.argsv[1:]
    main(keyword)
