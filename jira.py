"""

This utility will use the Jira REST API to download a project's
statuses, issues, and historical changelog data to be used in a
Kanban metrics analysis (see analysis.py)

"""
import argparse
import csv
import json
import logging
import os

import requests
from requests.auth import HTTPBasicAuth

import requests_cache
requests_cache.install_cache('jiracache', backend='sqlite', expire_after=24*60*60)

logger = logging.getLogger(__name__)


class Client:
    """ simple wrapper for client data """
    domain = ''
    email = ''
    apikey = ''

    def __init__(self, domain, email='', apikey=''):
        self.domain = domain
        self.email = email
        self.apikey = apikey

    def url(self, path):
        """ return a url prefixed with the domain """
        return self.domain + path

    def auth(self):
        """ return an auth object with the email and apikey """
        return HTTPBasicAuth(self.email, self.apikey)

    def headers(self):
        """ return the basic headers to send with requests """
        return {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

def fetch_status_categories_all(client):
    """ get all status categories """
    response = requests.get(client.url('/rest/api/3/statuscategory'),
                            auth=client.auth(), headers=client.headers())
    if response.status_code != 200:
        logging.warning('could not fetch status categories')
        return {}
    return json.loads(response.text)

def fetch_statuses_all(client):
    """ get all statuses """
    response = requests.get(client.url('/rest/api/3/status'),
                            auth=client.auth(), headers=client.headers())
    if response.status_code != 200:
        logging.warning('could not fetch statuses')
        return {}
    return json.loads(response.text)

def fetch_statuses_by_project(client, project_key):
    """ get all statuses in a project """
    response = requests.get(client.url('/rest/api/3/project/{}/statuses'.format(project_key)), auth=client.auth(), headers=client.headers())
    if response.status_code != 200:
        logging.warning('could not fetch project {} statuses'.format(project_key))
        return {}
    return json.loads(response.text)

def fetch_project(client, project_key):
    """ get a project """
    response = requests.get(client.url('/rest/api/3/project/{}'.format(project_key)), auth=client.auth(), headers=client.headers())
    if response.status_code != 200:
        logging.warning('could not fetch project {}'.format(project_key))
        return {}
    return json.loads(response.text)

def fetch_changelog(client, issue_id, start=0, limit=10):
    """ get an issue changelog """
    params={'startAt': start, 'maxResults': limit}
    response = requests.request('GET', client.url('/rest/api/3/issue/{}/changelog'.format(issue_id)), params=params, auth=client.auth(), headers=client.headers())
    if response.status_code != 200:
        logging.warning('could not fetch changelog for issue {}'.format(issue_id))
        return {}
    return json.loads(response.text)

def yield_changelog_all(client, issue_id, batch=100):
    """ iterate through all changelog items in an issue """
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

def fetch_issues(client, project_key, since='2020-01-01', start=0, limit=1000, custom_fields=None, updates_only=False, use_get=False):
    """ get all issues matching the filters in a project """
    jql = 'project = {} AND created >= "{}" ORDER BY created ASC'.format(project_key, since)
    
    if updates_only:
        jql = 'project = {} AND updated >= "{}" ORDER BY created ASC'.format(project_key, since)
    
    fields = [
        'parent',
        'summary',
        'status',
        'issuetype',
        'created',
        'updated'
    ]
    
    if custom_fields:
        fields = fields + custom_fields
    
    payload = {
      'jql': jql,
      'fieldsByKeys': False,
      'fields': fields,
      'expand':'names',
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

def yield_issues_all(client, project_key, since='2020-01-01', batch=100, custom_fields=None, updates_only=False, use_get=False):
    """ iterate through all issues in a project matching the filter """
    issues_count = fetch_issues(client, project_key, since=since, start=0, limit=0, custom_fields=custom_fields, updates_only=updates_only, use_get=use_get)
    total = issues_count.get('total', 0)
    fetched = 0
    while fetched < total:
        j = fetch_issues(client, project_key, since=since, start=fetched, limit=batch, custom_fields=custom_fields, updates_only=updates_only, use_get=use_get)
        if not j:
            break
        k = j.get('issues', [])
        if not k:
            break
        for result in k:
            yield result
            fetched += 1

def fetch(client, project_key, since='2020-01-01', custom_fields=None, updates_only=False):
    """ get all issue and changelog information for a project """
    logging.info('fetching project {} since {}...'.format(project_key, since))
    
    # get high level information fresh every time
    with requests_cache.disabled():
        categories = fetch_status_categories_all(client)
        statuses = fetch_statuses_all(client)
        project = fetch_project(client, project_key)
        project_statuses = fetch_statuses_by_project(client, project_key) 

    # compute lookup tables
    categories_by_category_id = {}
    for category in categories:
        categories_by_category_id[category.get('id')] = category

    status_categories_by_status_id = {}
    for status in statuses:
        status_categories_by_status_id[int(status.get('id'))] = categories_by_category_id[status.get('statusCategory', {}).get('id')]

    # fetch issues!
    issues = yield_issues_all(client, project_key, since=since, custom_fields=custom_fields, updates_only=updates_only, use_get=True)
    
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
        
        suffix = {}
        if custom_fields:
            suffix = {k: issue.get('fields', {}).get(k) for k in custom_fields}
        
        changelog = yield_changelog_all(client, issue_id)
        has_status = False
        for changeset in changelog:
            logging.info('fetching changelog for issue {}...'.format(issue.get('key')))
            
            for record in changeset.get('items', []):
                if record.get('field') == 'status':
                    from_category = status_categories_by_status_id.get(int(record.get('from')), {})
                    to_category = status_categories_by_status_id.get(int(record.get('to')), {})
                    
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
                    row.update(suffix)
                    
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
            row.update(suffix)
            yield row


def generate_csv(client, csv_file, project_key, since='2020-01-01', custom_fields=None, custom_field_names=None, updates_only=False, write_header=False, anonymize=False):
    """
    iterate through all issues in a project matching the filter criteria and write it to csv 
    
    optionally, fetch custom fields and map them to custom column names in the csv
    
    """
    import datetime
    import dateutil.parser
    import pytz
    
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
    
    custom_field_map = {}
    if custom_fields:
        if custom_field_names:
            custom_field_map = dict(zip(custom_fields, custom_field_names))
            fieldnames.extend(custom_field_names)
        else:
            fieldnames.extend(custom_fields)    
        
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    if write_header:
        writer.writeheader()
    
    count = 0
    for record in fetch(client, project_key, since=since, custom_fields=custom_fields, updates_only=updates_only):
        for key, value in record.items():
            # ensure ISO datetime strings with TZ offsets to ISO datetime strings in UTC
            if 'date' in key and value and not isinstance(value, datetime.datetime):
                value = dateutil.parser.parse(value)
                value = value.astimezone(pytz.UTC)
                record[key] = value.isoformat()

        if anonymize:
            record['issue_key'] = record['issue_key'].replace(record['project_key'], 'PRJ') 
            record['project_key'] = 'PRJ'
            record['issue_title'] = None
            
        if custom_field_map:
            for key, value in custom_field_map.items():
                if key not in record:
                    continue
                record[value] = record[key]
                del record[key]
                
        writer.writerow(record)
        count += 1
    
    logging.info('{} records written'.format(count))
        
        
def main():
    """ parse the command line arguments """
    domain = os.getenv('JIRA_DOMAIN', '')
    email  = os.getenv('JIRA_EMAIL', '')
    apikey = os.getenv('JIRA_APIKEY', '')

    parser = argparse.ArgumentParser(description='Extract issue changelog data from a Jira Project')
    parser.add_argument('project', help='Jira project from which to extract issues')
    parser.add_argument('since', help='Date from which to start extracting issues (yyyy-mm-dd)')
    parser.add_argument('--updates-only', action='store_true', help='''
        When passed, instead of extracting issues created since the since argument,
        only issues *updated* since the since argument will be extracted.''')
    parser.add_argument('--append', action='store_true', help='Append to the output file instead of overwriting it.')
    parser.add_argument('--anonymize', action='store_true', help='Anonymize the data output (no issue titles, project keys, etc).')
    parser.add_argument('-d', '--domain', default=domain, help='Jira project domain url (i.e., https://company.atlassian.net). Can also be provided via JIRA_DOMAIN environment variable.')
    parser.add_argument('-e', '--email',  default=email,  help='Jira user email address for authentication. Can also be provided via JIRA_EMAIL environment variable.')
    parser.add_argument('-k', '--apikey', default=apikey, help='Jira user api key for authentication. Can also be provided via JIRA_APIKEY environment variable.')
    parser.add_argument('-o', '--output', default='out.csv', help='File to store the csv output.')
    parser.add_argument('-q', '--quiet', action='store_true', help='Be quiet and only output warnings to console.')
    
    parser.add_argument('-f', '--field', metavar='FIELD_ID', action='append', help='Include one or more custom fields in the query by id.')
    parser.add_argument('-n', '--name', metavar='FIELD_NAME', action='append', help='Corresponding output column names for each custom field.')
    
    args = parser.parse_args()
    
    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
        
    if not all((args.domain, args.email, args.apikey)):
        parser.error("""The JIRA_DOMAIN, JIRA_EMAIL, and JIRA_APIKEY environment variables """
                     """must be set or provided via the -d -e -k command line flags.""")
        return
    
    logging.info('connecting to {} with {} email...'.format(args.domain, args.email))
    
    client = Client(args.domain, email=args.email, apikey=args.apikey)
    
    mode = 'a' if args.append else 'w' 
    
    custom_fields = [k if k.startswith('customfield') else 'customfield_{}'.format(k) for k in args.field] if args.field else []
    custom_field_names = list(args.name) + custom_fields[len(args.name):]
    
    with open(args.output, mode, newline='') as csv_file:
        logging.info('{} opened for writing (mode: {})...'.format(args.output, mode))
        generate_csv(client, csv_file, args.project,
                     since=args.since,
                     custom_fields=custom_fields,
                     custom_field_names=custom_field_names,
                     updates_only=args.updates_only,
                     write_header=not args.append,
                     anonymize=args.anonymize)


if __name__ == '__main__':
    main()
