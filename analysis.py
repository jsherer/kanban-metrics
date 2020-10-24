"""

This utility will use the a project's historical changelog data to
compute Kanban metrics analysis. (see jira.py)

"""
from __future__ import print_function

import argparse
import collections
import logging
import sys

import lifelines
import matplotlib
import matplotlib.pyplot
import numpy
import pandas
import seaborn

from pandas.plotting import register_matplotlib_converters

darkgrey = '#3A3A3A'
lightgrey = '#414141'
logger = logging.getLogger(__file__)

class AnalysisException(Exception):
    pass


def init():
    """ initialize basic plotting styles """
    register_matplotlib_converters()
    matplotlib.pyplot.style.use('fivethirtyeight')
    matplotlib.pyplot.rcParams['axes.labelsize'] = 14
    matplotlib.pyplot.rcParams['lines.linewidth'] = 1.5


def read_data(path, omit=None, points_field=None, since='', until=''):
    """
    read csv changelog data with necessary fields:
    
    * issue_id - unique numeric id for this issue
    * issue_key - unique textual key for this issue
    * issue_type_name - category of issue type
    * issue_created_date - when the issue was created
    * changelog_id - unique id for this particular change for this issue
    * status_change_date - when the change was made
    * status_from_name - from which status the issue was changed (optional)
    * status_to_name - to which status the issue was changed
    * status_from_category_name - from which status category the issue was changed (optional)
    * status_to_category_name - to which status category the issue was changed
    
    """
    OMIT_ISSUE_TYPES = set(omit) if omit else None
    STORY_POINT_FIELD = points_field
    
    logger.info(f'Opening input file for reading...')

    data = pandas.read_csv(path)
    
    required_fields = [
        'issue_id',
        'issue_key',
        'issue_type_name',
        'issue_created_date',
        'changelog_id',
        'status_change_date',
        'status_from_name',
        'status_to_name',
        'status_from_category_name',
        'status_to_category_name',
    ]
    
    # check for missing fields
    missing_fields = []
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
    if missing_fields:
        raise AnalysisException(f'Required fields `{", ".join(missing_fields)}` missing from the dataset')
    
    # parse the datetimes to utc and then localize them to naive datetimes so _all_ date processing in pandas is naive in UTC
    data['issue_created_date'] = data['issue_created_date'].apply(pandas.to_datetime, utc=True).dt.tz_localize(None)
    data['status_change_date'] = data['status_change_date'].apply(pandas.to_datetime, utc=True).dt.tz_localize(None)

    # let's check to make sure the data is sorted correctly by issue_id and status_change_date
    data = data.sort_values(['issue_id', 'status_change_date'])
    
    # let's drop duplicates based on issue_id and changelog_id
    n1 = len(data)
    logger.info(f'-> {n1} changelog items read')
    data = data.drop_duplicates(subset=['issue_id', 'changelog_id'], keep='first')
    
    # count how many changelog items were duplicates
    n2 = len(data)
    dupes = n1-n2
    logger.info(f'-> {dupes} changelog items removed as duplicate')

    # filter out Epic specific data
    if OMIT_ISSUE_TYPES:
        data = data[~data['issue_type_name'].isin(OMIT_ISSUE_TYPES)]
        n3 = len(data)
        omitted = n2-n3
        logger.info(f'-> {omitted} changelog items omitted')

    # filter out issues before since date and after until
    if since:
        data = data[data['issue_created_date'] >= since]
    if until:
        data = data[data['issue_created_date'] < until]
    
    # count how many changelog items were filtered
    n3 = len(data)
    filtered = n2-n3
    logger.info(f'-> {filtered} changelog items filtered')
    
    # rename the story point field name to "issue_points"
    if STORY_POINT_FIELD:
        data = data.rename(columns={STORY_POINT_FIELD:'issue_points'})
    if 'issue_points' not in data:
        data['issue_points'] = 1
    
    min_date = data['issue_created_date'].min().strftime('%Y-%m-%d')
    max_date = data['issue_created_date'].max().strftime('%Y-%m-%d')

    logger.info(f'-> {n3} changelog items remain created from {min_date} to {max_date}')
    logger.info('---')

    return data, dupes, filtered


def process_flow_data(data, since='', until=''):
    """ process cumulative flow statuses from changelog data """
    STATUS_ORDER = ['Prioritized', 'In Preparation', 'In Progress', 'Review', 'Accepted', 'Deployed']

    if data.empty:
        logger.warning('Warning: Data for flow analysis is empty')
        return
    
    if not since or not until:
        raise AnalysisException('Flow analysis requires both `since` and `until` dates for processing')
    
    dates = pandas.date_range(start=since, end=until, closed='left', freq='D')
    
    flow_data = data.copy().reset_index()
    flow_data = flow_data.sort_values(['status_change_date'])

    statuses = set(flow_data['status_from_name']) | set(flow_data['status_to_name'])
    if numpy.nan in statuses:
        statuses.remove(numpy.nan)
        
    f = pandas.DataFrame(columns=['date'] + list(statuses))

    last_counter = None

    for date in dates:
        tomorrow = date + pandas.Timedelta(days=1)
        date_changes = flow_data
        date_changes = date_changes[date_changes['status_change_date'] >= date]
        date_changes = date_changes[date_changes['status_change_date'] < tomorrow]

        if last_counter:
            counter = last_counter
        else:
            counter = collections.Counter()
        for item in date_changes['status_from_name']:
            if counter[item] > 0:
                counter[item] -= 1
        for item in date_changes['status_to_name']:
            counter[item] += 1
        
        row = dict(counter)
        row['date'] = date
        f = f.append(row, ignore_index=True)
        
        last_counter = counter

    f = f.fillna(0)
    if f.empty:
        return f
    
    f['date'] = f['date'].dt.normalize()
    f['date'] = f['date'].dt.date
    ending_status = STATUS_ORDER[-1]
    status_columns = list(reversed(STATUS_ORDER))
    flow_columns = ['date'] + [status for status in status_columns if status in f]
    flow = f[flow_columns]

    return flow


def process_issue_data(data):
    """ process aggregate issue data from changelog data, calculating cycle and lead times """
    if data.empty:
        logger.warning('Warning: Data for issue analysis is empty')
        return
    
    issues = collections.defaultdict(list)
    issue_ids = dict()
    issue_keys = dict()
    issue_types = dict()
    issue_points = dict()
    for row, item in data.iterrows():
        issues[item.issue_id].append(item)
        issue_types[item.issue_id] = item.issue_type_name
        issue_ids[item.issue_key] = item.issue_id
        issue_keys[item.issue_id] = item.issue_key
        issue_points[item.issue_id] = item.issue_points


    # What we want to be able to do at this point is to know the total time an issue spends in the "In Progress" state. We could take a look at all of the state changes and compute the sum of the time residing in the "In Progess" state. An alternative that is easier to compute (with less accuracy) is to track when the issue was first created, when it first moved into work in progress, and when it finally completed, ignoring other state transitions in between.
    # 
    # To do this without having to map each status, we use the Jira "status_category_name", which is an ENUM:
    # 
    # * To Do
    # * In Progress
    # * Done

    issue_statuses = collections.defaultdict(dict)

    for issue_id, issue in issues.items():
        for update in issue:
            # learn when the issue was first created
            if not issue_statuses[issue_id].get('first_created'):
                issue_statuses[issue_id]['first_created'] = update.issue_created_date
            issue_statuses[issue_id]['first_created'] = min(issue_statuses[issue_id]['first_created'], update.issue_created_date)

            # learn when the issue was first moved to in progress
            if update.status_to_category_name == 'In Progress':
                if not issue_statuses[issue_id].get('first_in_progress'):
                    issue_statuses[issue_id]['first_in_progress'] = update.status_change_date
                issue_statuses[issue_id]['first_in_progress'] = min(issue_statuses[issue_id]['first_in_progress'], update.status_change_date)

            # learn when the issue was finally moved to completion
            if update.status_to_category_name == 'Complete' or update.status_to_category_name == 'Done':
                if not issue_statuses[issue_id].get('last_complete'):
                    issue_statuses[issue_id]['last_complete'] = update.status_change_date
                issue_statuses[issue_id]['last_complete'] = max(issue_statuses[issue_id]['last_complete'], update.status_change_date)

            issue_statuses[issue_id]['last_update'] = update


    # Finally, we create a new data set of each issue with the dates when the state changes happened. We also compute the lead and cycle times of each issue.
    issue_data = pandas.DataFrame(columns=[
        'issue_key',
        'issue_type',
        'issue_points',
        'new',
        'new_day',
        'in_progress',
        'in_progress_day',
        'complete',
        'complete_day',
        'lead_time',
        'lead_time_days',
        'cycle_time',
        'cycle_time_days',
    ])

    for issue_id in issue_statuses:
        new = issue_statuses[issue_id].get('first_created')
        in_progress = issue_statuses[issue_id].get('first_in_progress')
        complete = issue_statuses[issue_id].get('last_complete')

        # since numpy uses naive datetimes, let's make these dates naive
        if new:
            new = new.replace(tzinfo=None)
        if in_progress:
            in_progress = in_progress.replace(tzinfo=None)
        if complete:
            complete = complete.replace(tzinfo=None)

        if complete:
            lead_time = complete - new

            if in_progress:
                cycle_time = complete - in_progress

                # adjust for weekends in the cycle_time
                weekend_days = numpy.busday_count(in_progress.date(), complete.date(), weekmask='Sat Sun')
                cycle_time -= pandas.Timedelta(days=weekend_days)

            else:
                cycle_time = pandas.Timedelta(days=0)
        else:
            lead_time = pandas.Timedelta(days=0)
            cycle_time = pandas.Timedelta(days=0)

        # restore UTC dates
        # if new:
        #     new = pandas.to_datetime(new, utc=True)
        # if in_progress:
        #     in_progress = pandas.to_datetime(in_progress, utc=True)
        # if complete:
        #     complete = pandas.to_datetime(complete, utc=True)
            
        issue_data = issue_data.append({
            'issue_key': issue_keys.get(issue_id),
            'issue_type': issue_types.get(issue_id),
            'issue_points': issue_points.get(issue_id),
            'new': new,
            'new_day': None,
            'in_progress': in_progress,
            'in_progress_day': None,
            'complete': complete,
            'complete_day': None,
            'lead_time': lead_time,
            'lead_time_days': None,
            'cycle_time': cycle_time,
            'cycle_time_days': None,
        }, ignore_index=True)

    # truncate days to omit time
    issue_data['new_day'] = issue_data['new'].values.astype('<M8[D]')
    #issue_data['new_day'] = issue_data['new_day'].apply(pandas.to_datetime, utc=True)
    
    issue_data['in_progress_day'] = issue_data['in_progress'].values.astype('<M8[D]')
    #issue_data['in_progress_day'] = issue_data['in_progress_day'].apply(pandas.to_datetime, utc=True)
    
    issue_data['complete_day'] = issue_data['complete'].values.astype('<M8[D]')
    #issue_data['complete_day'] = issue_data['complete_day'].apply(pandas.to_datetime, utc=True)

    # add column for lead time represented as days
    issue_data['lead_time_days'] = issue_data['lead_time'] / pandas.to_timedelta(1, unit='D')
    # round lead time less than 1 hour to zero
    issue_data.loc[issue_data['lead_time_days'] < 1/24.0, 'lead_time_days'] = 0

    # add column for cycle time represented as days
    issue_data['cycle_time_days'] = issue_data['cycle_time'] / pandas.to_timedelta(1, unit='D')
    # round cycle time less than 1 hour to zero
    issue_data.loc[issue_data['cycle_time_days'] < 1/24.0, 'cycle_time_days'] = 0

    # add column for the last statuses of this issue
    issue_data['last_issue_status'] = [issue_statuses[issue_ids[key]].get('last_update', {}).get('status_to_name') for key in issue_data['issue_key']]
    issue_data['last_issue_status_category'] = [issue_statuses[issue_ids[key]].get('last_update', {}).get('status_to_category_name') for key in issue_data['issue_key']]
    
    extra = (
        issues,
        issue_ids,
        issue_keys,
        issue_types,
        issue_points,
        issue_statuses,
    )

    return issue_data, extra


def process_cycle_data(issue_data, since='', until=''):
    """ process cycle time data from issue data """
    if issue_data.empty:
        logger.warning('Warning: Data for cycle analysis is empty')
        return
    
    cycle_data = issue_data.copy().reset_index()
    cycle_data = cycle_data.sort_values(['complete'])

    if since:
        cycle_data = cycle_data[cycle_data['complete_day'] >= pandas.to_datetime(since)]
    
    if until:
        cycle_data = cycle_data[cycle_data['complete_day'] < pandas.to_datetime(until)]

    # drop issues with a cycle time less than 1 hour
    cycle_data = cycle_data[cycle_data['cycle_time_days'] > (1/24.0)]

    cycle_data['Moving Average (10 days)'] = cycle_data['cycle_time_days'].rolling(window=10).mean()
    cycle_data['Moving Standard Deviation (10 days)'] = cycle_data['cycle_time_days'].rolling(window=10).std()
    cycle_data['Average'] = cycle_data['cycle_time_days'].mean()
    cycle_data['Standard Deviation'] = cycle_data['cycle_time_days'].std()
    
    return cycle_data


def process_throughput_data(issue_data, since='', until=''):
    """ process throughput data from issue data """
    if issue_data.empty:
        logger.warning('Warning: Data for throughput analysis is empty')
        return
    
    throughput_data = issue_data.copy()
    throughput_data = throughput_data.sort_values(['complete'])
    
    if since:
        throughput_data = throughput_data[throughput_data['complete_day'] >= pandas.to_datetime(since)]

    if until:
        throughput_data = throughput_data[throughput_data['complete_day'] < pandas.to_datetime(until)]
        
    points_data = pandas.pivot_table(throughput_data, values='issue_points', index='complete_day', aggfunc=numpy.sum)

    throughput = pandas.crosstab(throughput_data.complete_day, issue_data.issue_type, colnames=[None]).reset_index()
    
    date_range = pandas.date_range(
        start=throughput.complete_day.min(),
        end=throughput.complete_day.max(), freq='B'
    )
    
    cols = set(throughput.columns)
    if 'complete_day' in cols:
        cols.remove('complete_day')

    throughput['Throughput'] = 0
    for col in cols:
        throughput['Throughput'] += throughput[col]   

    throughput = throughput.set_index('complete_day')
    throughput['Velocity'] = points_data['issue_points']

    throughput = throughput.reindex(date_range).fillna(0).astype(int).rename_axis('Date')
    
    throughput['Moving Average (10 days)'] = throughput['Throughput'].rolling(window=10).mean()
    throughput['Moving Standard Deviation (10 days)'] = throughput['Throughput'].rolling(window=10).std()
    throughput['Average'] = throughput['Throughput'].mean()
    throughput['Standard Deviation'] = throughput['Throughput'].std()

    throughput_per_week = pandas.DataFrame(
        throughput['Throughput'].resample('W-Mon').sum()
    ).reset_index()

    throughput_per_week['Moving Average (4 weeks)'] = throughput_per_week['Throughput'].rolling(window=4).mean()
    throughput_per_week['Moving Standard Deviation (4 weeks)'] = throughput_per_week['Throughput'].rolling(window=4).std()
    throughput_per_week['Average'] = throughput_per_week['Throughput'].mean()
    throughput_per_week['Standard Deviation'] = throughput_per_week['Throughput'].std()

    return throughput, throughput_per_week


def process_wip_data(issue_data, since='', until=''):
    """ process wip average data from issue data """
    if issue_data.empty:
        logger.warning('Warning: Data for wip analysis is empty')
        return
    
    WIP_OPEN_AT_START=0 # change this if you want to push up the baseline

    wip_data = issue_data[issue_data['in_progress_day'].notnull()]
    wip_data = wip_data[wip_data['last_issue_status_category'] != 'To Do']
    wip_data = wip_data.sort_values(['in_progress'])

    date_range = pandas.date_range(start=since, end=until, closed='left', freq='D')

    wip = pandas.DataFrame(columns=['date', 'WIP'])

    for date in date_range:
        tomorrow = date + pandas.Timedelta(days=1)
        date_changes = wip_data
        date_changes = date_changes[date_changes['in_progress_day'] <= date]
        date_changes = date_changes[(date_changes['complete_day'].isnull()) | (date_changes['complete_day'] > date)]
        
        row = dict()
        row['date'] = date
        row['WIP'] = WIP_OPEN_AT_START + len(date_changes)
        wip = wip.append(row, ignore_index=True)
        
    wip = wip.set_index('date')
    wip = wip.reindex(date_range).fillna(0).astype(int).rename_axis('Date')
        
    wip['Moving Average (10 days)'] = wip['WIP'].rolling(window=10).mean()
    wip['Moving Standard Deviation (10 days)'] = wip['WIP'].rolling(window=10).std()
    wip['Average'] = wip['WIP'].mean()
    wip['Standard Deviation'] = wip['WIP'].std()
    
    # resample to also provide how much wip we have at the end of each week
    wip_per_week = pandas.DataFrame(
        wip['WIP'].resample('W-Mon').last()
    ).reset_index()

    wip_per_week['Moving Average (4 weeks)'] = wip_per_week['WIP'].rolling(window=4).mean()
    wip_per_week['Moving Standard Deviation (4 weeks)'] = wip_per_week['WIP'].rolling(window=4).std()
    wip_per_week['Average'] = wip_per_week['WIP'].mean()
    wip_per_week['Standard Deviation'] = wip_per_week['WIP'].std()

    return wip, wip_per_week


def process_wip_age_data(issue_data, since='', until=''):
    """ process wip age data from issue data """
    if issue_data.empty:
        logger.warning('Warning: Data for wip age analysis is empty')
        return
    
    age_data = issue_data[issue_data['in_progress_day'].notnull()]
    
    if since:
        age_data = age_data[age_data['in_progress_day'] >= pandas.to_datetime(since)]
    
    if until:
        age_data = age_data[age_data['in_progress_day'] < pandas.to_datetime(until)]
    
    age_data = age_data[(age_data['complete_day'].isnull()) | (age_data['complete_day'] < pandas.to_datetime(until))]
    age_data = age_data[age_data['last_issue_status_category'] != 'To Do']
    age_data = age_data.sort_values(['in_progress'])

    today = pandas.to_datetime(until)
    
    age_data['Age'] = (today - age_data['in_progress']) / pandas.to_timedelta(1, unit='D')
    age_data['Average'] = age_data['Age'].mean()
    age_data['Standard Deviation'] = age_data['Age'].std()
    age_data['P50']  = age_data['Age'].quantile(0.5)
    age_data['P75']  = age_data['Age'].quantile(0.75)
    age_data['P85']  = age_data['Age'].quantile(0.85)
    age_data['P95']  = age_data['Age'].quantile(0.95)

    return age_data


def forecast_montecarlo_how_long_items(throughput_data, items=10, simulations=10000, window=90):
    """ forecast number of days it will take to complete n number of items """
    if throughput_data.empty:
        logger.warning('Warning: Data for Montecarlo analysis is empty')
        return
    
    SIMULATION_ITEMS = items
    SIMULATIONS = simulations
    LAST_DAYS = window

    def simulate_days(data, scope):
        days = 0
        total = 0
        while total <= scope:
            total += data.sample(n=1).iloc[0]['Throughput']
            days += 1
        return days 

    dataset = throughput_data[['Throughput']].tail(LAST_DAYS).reset_index(drop=True)
    count = len(dataset)
    if count < window:
        logger.warning(f'Warning: Montecarlo window ({window}) is larger than throughput dataset ({count})')
    
    samples = [simulate_days(dataset, SIMULATION_ITEMS)
            for i in range(SIMULATIONS)]
    samples = pandas.DataFrame(samples, columns=['Days'])
    distribution_how_long = samples.groupby(['Days']).size().reset_index(name='Frequency')
    distribution_how_long = distribution_how_long.sort_index(ascending=False)
    distribution_how_long['Probability'] = 100 - 100 *     distribution_how_long.Frequency.cumsum()/distribution_how_long.Frequency.sum()

    return distribution_how_long, samples


def forecast_montecarlo_how_many_items(throughput_data, days=10, simulations=10000, window=90):
    """ forecast number of items completed in n days """
    if throughput_data.empty:
        logger.warning('Warning: Data for Montecarlo analysis is empty')
        return
    
    SIMULATION_DAYS = days
    SIMULATIONS = simulations
    LAST_DAYS = window
    
    dataset = throughput_data[['Throughput']].tail(LAST_DAYS).reset_index(drop=True)
    count = len(dataset)
    if count < window:
        logger.warning(f'Warning: Montecarlo window ({window}) is larger than throughput dataset ({count})')
        
    samples = [dataset.sample(n=SIMULATION_DAYS, replace=True).sum()['Throughput'] for i in range(SIMULATIONS)]
    samples = pandas.DataFrame(samples, columns=['Items'])
    distribution_how = samples.groupby(['Items']).size().reset_index(name='Frequency')
    distribution_how = distribution_how.sort_index(ascending=False)
    distribution_how['Probability'] = 100 *     distribution_how.Frequency.cumsum()/distribution_how.Frequency.sum()

    return distribution_how, samples


def forecast_montecarlo_how_long_points(throughput_data, points=10, simulations=10000, window=90):
    """ forecast number of days it will take to complete n number of points """
    if throughput_data.empty:
        logger.warning('Warning: Data for Montecarlo analysis is empty')
        return
    
    SIMULATION_ITEMS = points
    SIMULATIONS = simulations
    LAST_DAYS = window
    
    if (throughput_data['Velocity']/throughput_data['Throughput']).max() == 1:
        logger.warning(f'Warning: All velocity data is equal. Did you load data with points fields?')

    def simulate_days(data, scope):
        days = 0
        total = 0
        while total <= scope:
            total += data.sample(n=1).iloc[0]['Velocity']
            days += 1
        return days 

    dataset = throughput_data[['Velocity']].tail(LAST_DAYS).reset_index(drop=True)
    
    count = len(dataset)
    if count < window:
        logger.warning(f'Warning: Montecarlo window ({window}) is larger than velocity dataset ({count})')
    
    samples = [simulate_days(dataset, SIMULATION_ITEMS)
            for i in range(SIMULATIONS)]
    samples = pandas.DataFrame(samples, columns=['Days'])
    distribution_how_long = samples.groupby(['Days']).size().reset_index(name='Frequency')
    distribution_how_long = distribution_how_long.sort_index(ascending=False)
    distribution_how_long['Probability'] = 100 - 100 *     distribution_how_long.Frequency.cumsum()/distribution_how_long.Frequency.sum()

    return distribution_how_long, samples


def forecast_montecarlo_how_many_points(throughput_data, days=10, simulations=10000, window=90):
    """ forecast number of points completed in n days """
    if throughput_data.empty:
        logger.warning('Warning: Data for Montecarlo analysis is empty')
        return
    
    SIMULATION_DAYS = days
    SIMULATIONS = simulations
    LAST_DAYS = window

    if (throughput_data['Velocity']/throughput_data['Throughput']).max() == 1:
        logger.warning(f'Warning: All velocity data is equal. Did you load data with points fields?')

    dataset = throughput_data[['Velocity']].tail(LAST_DAYS).reset_index(drop=True)
    count = len(dataset)
    if count < window:
        logger.warning(f'Warning: Montecarlo window ({window}) is larger than velocity dataset ({count})')
    
    samples = [dataset.sample(n=SIMULATION_DAYS, replace=True).sum()['Velocity'] for i in range(SIMULATIONS)]
    samples = pandas.DataFrame(samples, columns=['Points'])
    distribution_how = samples.groupby(['Points']).size().reset_index(name='Frequency')
    distribution_how = distribution_how.sort_index(ascending=False)
    distribution_how['Probability'] = 100 *     distribution_how.Frequency.cumsum()/distribution_how.Frequency.sum()

    return distribution_how, samples

def run(args):    
    data, dupes, filtered = read_data(args.file, omit=args.omit, since=args.since, until=args.until)
    if data.empty:
        logger.warning('Warning: Data for analysis is empty')
        return
    
    if args.command == 'summary':
        i, _ = process_issue_data(data)
        
        # current cycle time
        c = process_cycle_data(i, since=args.since, until=args.until)
        
        # current throughput
        t, tw = process_throughput_data(i, since=args.since, until=args.until)
        
        # current wip
        w, ww = process_wip_data(i, since=args.since, until=args.until)
        a = process_wip_age_data(i, since=args.since, until=args.until)

        args.output.writelines(f'{line}\n' for line in [
            'Cycle Time:', 
            '-> Average: %.2f' % c['Average'].iat[-1], 
            '-> Standard Deviation: %.2f' % c['Standard Deviation'].iat[-1], 
            '-> Moving Average (10 days): %.2f' % c['Moving Average (10 days)'].iat[-1], 
            '-> Moving Standard Deviation (10 days): %.2f' % c['Moving Standard Deviation (10 days)'].iat[-1], 
            '',
            'Throughput (Daily):', 
            '-> Average: %.2f' % t['Average'].iat[-1], 
            '-> Standard Deviation: %.2f' % t['Standard Deviation'].iat[-1], 
            '-> Moving Average (10 days): %.2f' % t['Moving Average (10 days)'].iat[-1], 
            '-> Moving Standard Deviation (10 days): %.2f' % t['Moving Standard Deviation (10 days)'].iat[-1], 
            '',
            'Throughput (Weekly):', 
            '-> Average: %.2f' % tw['Average'].iat[-1], 
            '-> Standard Deviation: %.2f' % tw['Standard Deviation'].iat[-1], 
            '-> Moving Average (4 weeks): %.2f' % tw['Moving Average (4 weeks)'].iat[-1], 
            '-> Moving Standard Deviation (4 weeks): %.2f' % tw['Moving Standard Deviation (4 weeks)'].iat[-1], 
            '',
            'Work in Progess (Daily):', 
            '-> Average: %.2f' % w['Average'].iat[-1], 
            '-> Standard Deviation: %.2f' % w['Standard Deviation'].iat[-1], 
            '-> Moving Average (10 days): %.2f' % w['Moving Average (10 days)'].iat[-1], 
            '-> Moving Standard Deviation (10 days): %.2f' % w['Moving Standard Deviation (10 days)'].iat[-1], 
            '',
            'Work in Progess (Weekly):', 
            '-> Average: %.2f' % ww['Average'].iat[-1], 
            '-> Standard Deviation: %.2f' % ww['Standard Deviation'].iat[-1], 
            '-> Moving Average (4 weeks): %.2f' % ww['Moving Average (4 weeks)'].iat[-1], 
            '-> Moving Standard Deviation (4 weeks): %.2f' % ww['Moving Standard Deviation (4 weeks)'].iat[-1], 
            '',
            f'Work in Progess Age (ending {args.until}):', 
            '-> Average: %.2f' % a['Average'].iat[-1], 
            '-> 50th Percentile: %.2f' % a['P50'].iat[-1], 
            '-> 75th Percentile: %.2f' % a['P75'].iat[-1], 
            '-> 85th Percentile: %.2f' % a['P85'].iat[-1], 
            '-> 95th Percentile: %.2f' % a['P95'].iat[-1],
        ])
    
    if args.command == 'flow':
        f = process_flow_data(data, since=args.since, until=args.until)
        print(f)
    
    if args.command == 'forecast-items': 
        # pre-req
        i, _ = process_issue_data(data)
        t, tw = process_throughput_data(i)
        
        # analysis
        if args.items:
            args.output.write(f'Montecarlo Forecast: How long to complete {args.items} items?\n')
            ml, s = forecast_montecarlo_how_long_items(t, items=args.items, simulations=args.simulations, window=args.window)
            for q in (0.25, 0.5, 0.75, 0.85, 0.95):
                args.output.write(f'-> {int(q*100)}th Percentile: %s Days\n' % s.Days.quantile(q))
            
        if args.days:
            args.output.write(f'Montecarlo Forecast: How many items can be completed in {args.days} days?\n')
            mh, s = forecast_montecarlo_how_many_items(t, days=args.days, simulations=args.simulations, window=args.window)
            for q in (0.25, 0.5, 0.75, 0.85, 0.95):
                args.output.write(f'-> {int(q*100)}th Percentile: %s Items\n' % s.Items.quantile(1-q))

        
    if args.command == 'forecast-points':
        i, _ = process_issue_data(data)
        t, tw = process_throughput_data(i)
        
        if args.points:
            args.output.write(f'Montecarlo Forecast: How long to complete {args.points} points?\n')
            mlp, s = forecast_montecarlo_how_long_points(t, items=args.points, simulations=args.simulations, window=args.window)
            for q in (0.25, 0.5, 0.75, 0.85, 0.95):
                args.output.write(f'-> {int(q*100)}th Percentile: %s Days\n' % s.Days.quantile(q))
        
        if args.days:
            args.output.write(f'Montecarlo Forecast: How many points can be completed in {args.days} days?\n')
            mhp, s = forecast_montecarlo_how_many_points(t, days=args.days, simulations=args.simulations, window=args.window)
            for q in (0.25, 0.5, 0.75, 0.85, 0.95):
                args.output.write(f'-> {int(q*100)}th Percentile: %s Points\n' % s.Points.quantile(1-q))


def main():
    parser = argparse.ArgumentParser(description='Analyze issue changelog data')
    parser.add_argument('-f', '--file', type=argparse.FileType('r'), default='-', help='Data file to analyze (default: stdin)')
    parser.add_argument('-o', '--output', type=argparse.FileType('w'), default='-', help='File to output results (default: stdout)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Be quiet and only output warnings to console.')
    
    parser.add_argument('--omit', action='append', help='Omit specific issue types from the analysis (e.g., "Epic", "Bug", etc)')
    parser.add_argument('--since', help='Only process issues created since date (format: YYYY-MM-DD)')
    parser.add_argument('--until', help='Only process issues created up until date (format: YYYY-MM-DD)')

    subparsers = parser.add_subparsers(dest='command')
    
    subparser_summary = subparsers.add_parser('summary')

    subparser_flow = subparsers.add_parser('flow')
    
    subparser_forecast_items = subparsers.add_parser('forecast-items')
    subparser_forecast_items.add_argument('--items', type=int, help='Number of items to predict answering the question "how many days to complete N items?"')
    subparser_forecast_items.add_argument('--days', type=int, help='Number of days to predict answering the question "how many items can be completed in N days?"')
    subparser_forecast_items.add_argument('--simulations', default=10000, help='Number of simulation iterations to run (default: 10000)')
    subparser_forecast_items.add_argument('--window', default=90, help='Window of historical data to use in the forecast (default: 90 days)')
    
    subparser_forecast_points = subparsers.add_parser('forecast-points')
    subparser_forecast_points.add_argument('--points', type=int, help='Number of points to predict answering the question "how many days to complete N points?"')
    subparser_forecast_points.add_argument('--days', type=int, help='Number of days to predict answering the question "how many points can be completed in N days?"')    
    subparser_forecast_points.add_argument('--simulations', default=10000, help='Number of simulation iterations to run (default: 10000)')
    subparser_forecast_points.add_argument('--window', default=90, help='Window of historical data to use in the forecast (default: 90 days)')

    args = parser.parse_args()
    
    if not args.quiet:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
    else:
        logging.basicConfig(level=logging.WARN, format='*** %(message)s ***')
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.output:
        args.output.reconfigure(line_buffering=True)
    
    try:
        init()
        run(args)
    except AnalysisException as e:
        logger.error('Error: %s', e)


if __name__ == '__main__':
    main()