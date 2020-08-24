"""

This utility will use the Jira REST API to download a project's
statuses, issues, and historical changelog data to be used in the 
Kanban metrics analysis instead of relying on a Looker query.

"""

import requests
from requests.auth import HTTPBasicAuth

import requests_cache
requests_cache.install_cache('jiracache', backend='sqlite', expire_after=24*60*60)

import argparse
import csv
import json
import logging
logger = logging.getLogger(__name__)


class Client:
    domain = ''
    email = ''
    apikey = ''
    
    def __init__(self, domain, email='', apikey=''):
        self.domain = domain
        self.email = email
        self.apikey = apikey
    
    def url(self, path):
        return self.domain + path
    
    def auth(self):
        return HTTPBasicAuth(self.email, self.apikey)
    
    def headers(self):
        return {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

def fetch_status_categories_all(client):
    response = requests.get(client.url('/rest/api/3/statuscategory'), auth=client.auth(), headers=client.headers())
    if response.status_code != 200:
        logging.warning('could not fetch status categories')
        return {}
    return json.loads(response.text)

def fetch_statuses_all(client):
    response = requests.get(client.url('/rest/api/3/status'), auth=client.auth(), headers=client.headers())
    if response.status_code != 200:
        logging.warning('could not fetch statuses')
        return {}
    return json.loads(response.text)

def fetch_statuses_by_project(client, project_key):
    response = requests.get(client.url('/rest/api/3/project/{}/statuses'.format(project_key)), auth=client.auth(), headers=client.headers())
    if response.status_code != 200:
        logging.warning('could not fetch project {} statuses'.format(project_key))
        return {}
    return json.loads(response.text)

def fetch_project(client, project_key):
    response = requests.get(client.url('/rest/api/3/project/{}'.format(project_key)), auth=client.auth(), headers=client.headers())
    if response.status_code != 200:
        logging.warning('could not fetch project {}'.format(project_key))
        return {}
    return json.loads(response.text)

def fetch_changelog(client, issue_id, start=0, limit=10):
    params={'startAt': start, 'maxResults': limit}
    response = requests.request('GET', client.url('/rest/api/3/issue/{}/changelog'.format(issue_id)), params=params, auth=client.auth(), headers=client.headers())
    if response.status_code != 200:
        logging.warning('could not fetch changelog for issue {}'.format(issue_id))
        return {}
    return json.loads(response.text)

def yield_changelog_all(client, issue_id, batch=100):
    starting_limit = 10
    changelog_count = fetch_changelog(client, issue_id, start=0, limit=starting_limit)
    total = changelog_count.get('total', 0)
    if total <= starting_limit:
        for result in changelog_count.get('values', []):
            yield result
    else:    
        fetched = 0
        while fetched < total:
            j = fetch_changelog(client, issue_id, start=fetched, limit=batch)
            if not j:
                break
            k = j.get('values', [])
            if not k:
                break
            for result in k:
                yield result
                fetched += 1

def fetch_issues(client, project_key, since='2020-01-01', start=0, limit=1000, updates_only=False, use_get=False):
    jql = 'project = {} AND created >= "{}" ORDER BY created ASC'.format(project_key, since)
    
    if updates_only:
        jql = 'project = {} AND updated >= "{}" ORDER BY created ASC'.format(project_key, since)
        
    payload = {
      'jql': jql,
      'fieldsByKeys': False,
      'fields': [
        'parent',
        'summary',
        'status',
        'issuetype',
        'created',
        'updated'
      ],
      'startAt': start,
      'maxResults': limit,
    }
    
    if use_get:
        response = requests.request(
           'GET',
           client.url('/rest/api/3/search'),
           params=payload,
           headers=client.headers(),
           auth=client.auth()
        )
    else:
        response = requests.request(
           'POST',
           client.url('/rest/api/3/search'),
           data=json.dumps(payload),
           headers=client.headers(),
           auth=client.auth()
        )
    
    if response.status_code != 200:
        logging.warning('could not fetch issues since {}'.format(since))
        return {}

    return json.loads(response.text)

def yield_issues_all(client, project_key, since='2020-01-01', batch=100, updates_only=False, use_get=False):
    issues_count = fetch_issues(client, project_key, since=since, start=0, limit=0, updates_only=updates_only, use_get=use_get)
    total = issues_count.get('total', 0)
    fetched = 0
    while fetched < total:
        j = fetch_issues(client, project_key, since=since, start=fetched, limit=batch, updates_only=updates_only, use_get=use_get)
        if not j:
            break
        k = j.get('issues', [])
        if not k:
            break
        for result in k:
            yield result
            fetched += 1

def fetch(client, project_key, since='2020-01-01', updates_only=False):
    logging.info('fetching project {} since {}...'.format(project_key, since))
    
    categories = fetch_status_categories_all(client)
    categories_by_category_id = {}
    for category in categories:
        categories_by_category_id[category.get('id')] = category
    
    statuses = fetch_statuses_all(client)
    project = fetch_project(client, project_key)
    project_statuses = fetch_statuses_by_project(client, project_key) 

    status_categories_by_status_id = {}
    for status in statuses:
        status_categories_by_status_id[int(status.get('id'))] = categories_by_category_id[status.get('statusCategory', {}).get('id')]
    
    issues = yield_issues_all(client, project_key, since=since, updates_only=updates_only, use_get=True)
    
    for issue in issues:
        logging.info('fetching issue {}...'.format(issue.get('key')))
        
        issue_id = issue.get('id')
    
        prefix = { 
            'project_id': project.get('id'),
            'project_key': project.get('key'),
            'issue_id': issue.get('id'),
            'issue_key': issue.get('key'),
            'issue_type_id': issue.get('fields', {}).get('issuetype', {}).get('id'),
            'issue_type_name': issue.get('fields', {}).get('issuetype', {}).get('name'),
            'issue_title': issue.get('fields', {}).get('summary'),
            'issue_created_date': issue.get('fields', {}).get('created'),
        }    
        
        changelog = yield_changelog_all(client, issue_id)
        has_status = False
        for changeset in changelog:
            logging.info('fetching changelog for issue {}...'.format(issue.get('key')))
            
            for record in changeset.get('items', []):
                if record.get('field') == 'status':
                    from_category = status_categories_by_status_id[int(record.get('from'))]
                    to_category = status_categories_by_status_id[int(record.get('to'))]
                    
                    row = dict(prefix)
                    row.update({
                        'changelog_id': changeset.get('id'),
                        'status_from_id': record.get('from'),
                        'status_from_name': record.get('fromString'),
                        'status_to_id': record.get('to'),
                        'status_to_name': record.get('toString'),
                        'status_from_category_name': from_category.get('name'),
                        'status_to_category_name': to_category.get('name'),
                        'status_change_date': changeset.get('created'),
                    })
                    
                    yield row
                    
                    has_status = True
                  
        # if we do not have a changelog status for this issue, we should emit a "new" status
        if not has_status:
            row = dict(prefix)
            row.update({
                'changelog_id': None,
                'status_from_id': None,
                'status_from_name': None,
                'status_to_id': None,
                'status_to_name': None,
                'status_from_category_name': None,
                'status_to_category_name': None, # new?
                'status_change_date': None
            })
            yield row


def generate_csv(client, csv_file, project_key, since='2020-01-01', updates_only=False):
    fieldnames = [
        'project_id',
        'project_key',
        'issue_id',
        'issue_key',
        'issue_type_id',
        'issue_type_name',
        'issue_title',
        'issue_created_date',
        'changelog_id',
        'status_from_id',
        'status_from_name',
        'status_to_id',
        'status_to_name',
        'status_from_category_name',
        'status_to_category_name',
        'status_change_date',
    ]

    import datetime
    import dateutil.parser
    import pytz

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    
    count = 0
    for record in fetch(client, project_key, since=since, updates_only=updates_only):
        for key, value in record.items():
            # ensure ISO datetime strings with TZ offsets to ISO datetime strings in UTC
            if 'date' in key and value and not isinstance(value, datetime.datetime):
                value = dateutil.parser.parse(value)
                value = value.astimezone(pytz.UTC)
                record[key] = value.isoformat()

        writer.writerow(record)
        count += 1
    
    logging.info('{} records written'.format(count))
        
        
def main():
    import argparse
    import os
    
    domain = os.getenv('JIRA_DOMAIN', '')
    email  = os.getenv('JIRA_EMAIL', '')
    apikey = os.getenv('JIRA_APIKEY', '')

    parser = argparse.ArgumentParser(description='Extract issue changelog data from a Jira Project')
    parser.add_argument('project', help='Jira project from which to extract issues')
    parser.add_argument('since', help='Date from which to start extracting issues (yyyy-mm-dd)')
    parser.add_argument('--updates-only', action='store_true', help='''
        When passed, instead of extracting issues created since the since argument,
        only issues *updated* since the since argument will be extracted.''')
    parser.add_argument('-d', '--domain', default=domain, help='Jira project domain url (i.e., https://company.atlassian.net). Can also be provided via JIRA_DOMAIN environment variable.')
    parser.add_argument('-e', '--email',  default=email,  help='Jira user email address for authentication. Can also be provided via JIRA_EMAIL environment variable.')
    parser.add_argument('-k', '--apikey', default=apikey, help='Jira user api key for authentication. Can also be provided via JIRA_APIKEY environment variable.')
    parser.add_argument('-o', '--output', default='out.csv', help='File to store the csv output.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose and output progress to console.')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    client = Client(args.domain, email=args.email, apikey=args.apikey)
    
    with open(args.output, 'w', newline='') as csv_file:
        logging.info('{} opened for writing...'.format(args.output))
        generate_csv(client, csv_file, args.project, updates_only=args.updates_only, since=args.since)


if __name__ == '__main__':
    main()