"""

This utility will use the a project's historical changelog data to
compute Kanban metrics analysis. (see jira.py)

"""
from __future__ import print_function

import argparse
import code
import collections
import functools
import logging

import lifelines
import matplotlib
import matplotlib.pyplot
import numpy
import pandas
import pingouin
import seaborn

from pandas.plotting import register_matplotlib_converters


logger = logging.getLogger(__file__)
if __name__ != '__main__':
    logging.basicConfig(level=logging.WARN)


class AnalysisException(Exception):
    pass


class Formatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = "%(message)s"
        else:
            self._style._fmt = "%(levelname)s: %(message)s"
        return super().format(record)


def init():
    """ initialize basic plotting styles """
    register_matplotlib_converters()
    matplotlib.pyplot.style.use('fivethirtyeight')
    matplotlib.pyplot.rcParams['axes.labelsize'] = 14
    matplotlib.pyplot.rcParams['lines.linewidth'] = 1.5


def read_data(path, exclude_types=None, since='', until=''):
    """
    read csv changelog data with necessary fields:

    * issue_id - unique numeric id for this issue
    * issue_key - unique textual key for this issue
    * issue_type_name - category of issue type
    * issue_created_date - when the issue was created
    * issue_points - how many points were assigned to this issue (optional)
    * changelog_id - unique id for this particular change for this issue
    * status_change_date - when the change was made
    * status_from_name - from which status (optional)
    * status_to_name - to which status
    * status_from_category_name - from which status category (optional)
    * status_to_category_name - to which status category

    """
    OMIT_ISSUE_TYPES = set(exclude_types) if exclude_types else None

    logger.info('Opening input file for reading...')

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
        logger.info(f'-> {omitted} changelog items excluded by type')

    # filter out issues before since date and after until
    if since:
        data = data[data['issue_created_date'] >= pandas.to_datetime(since)]
    if until:
        data = data[data['issue_created_date'] < pandas.to_datetime(until)]

    # count how many changelog items were filtered
    n3 = len(data)
    filtered = n2-n3
    logger.info(f'-> {filtered} changelog items filtered')

    # if issue_points does not exist, set them all to 1
    if 'issue_points' not in data:
        data['issue_points'] = 1

    if not data.empty:
        min_date = data['issue_created_date'].min().strftime('%Y-%m-%d')
        max_date = data['issue_created_date'].max().strftime('%Y-%m-%d')
        status_min_date = data['status_change_date'].min().strftime('%Y-%m-%d')
        status_max_date = data['status_change_date'].max().strftime('%Y-%m-%d')
        num_issues = data['issue_key'].nunique()
        logger.info(f'-> {n3} changelog items remain created from {min_date} to {max_date} with status changes from {status_min_date} to {status_max_date}')
        logger.info(f'-> {num_issues} unique work items loaded since {since}, ready for analysis')

    logger.info('---')

    return data, dupes, filtered


def process_flow_data(data, since='', until=''):
    """ process cumulative flow statuses from changelog data """

    if data.empty:
        logger.warning('Data for flow analysis is empty')
        return

    if not since or not until:
        raise AnalysisException('Flow analysis requires both `since` and `until` dates for processing')

    dates = pandas.date_range(start=since, end=until, closed='left', freq='D')

    flow_data = data.copy().reset_index()
    flow_data = flow_data.sort_values(['status_change_date'])

    statuses = set(flow_data['status_from_name']) | set(flow_data['status_to_name'])
    if numpy.nan in statuses:
        statuses.remove(numpy.nan)

    f = pandas.DataFrame(columns=['Date'] + list(statuses))

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
        row['Date'] = date
        f = f.append(row, ignore_index=True)

        last_counter = counter

    f = f.fillna(0)
    if f.empty:
        return f

    f['Date'] = f['Date'].dt.normalize()
    f['Date'] = f['Date'].dt.date

    f = f.set_index('Date')

    return f


def plot_flow_data(flow_data, status_columns=None):
    """ plot the cumulative flow diagram """
    STATUS_ORDER = [status for status in status_columns if status in flow_data]
    ending_status = STATUS_ORDER[-1]
    status_columns = list(reversed(STATUS_ORDER))
    flow_columns = ['Date'] + status_columns
    flow = flow_data.reset_index()[flow_columns]

    y_min = flow[ending_status].min()
    y_max = flow[STATUS_ORDER].max().sum()

    flow_agg = flow[flow_columns]
    x = flow_agg['Date'] = pandas.to_datetime(flow_agg['Date'])

    ys = []
    for status in reversed(STATUS_ORDER):
        lasty = ys[-1] if ys else 0
        y = flow_agg[status] = (flow_agg[status] + lasty).astype(float)
        ys.append(y)

    # melt the data to be able to be sent to lineplot
    xyz = pandas.melt(flow_agg, ['Date'])
    xyz['value'] = xyz['value'].astype(float)

    matplotlib.pyplot.figure(dpi=150, figsize=(15, 10))

    ax = seaborn.lineplot(x='Date', y='value', hue='variable', data=xyz)

    # create the area fills between lines
    lasty = 0
    for i, y in enumerate(ys):
        ax.fill_between(x, lasty, y, color=f'C{i}', alpha=0.7, interpolate=False)
        lasty = y

    ax.set_title("Cumulative Flow Since {}".format(flow_agg['Date'].min().strftime('%Y-%m-%d')), loc='left', fontdict={
                'fontsize': 18, 'fontweight': 'semibold'})

    ax.set_xlabel('Timeline')
    ax.set_ylabel('Items')

    tenth = (y_max-y_min)*0.1

    ax.set_ylim([y_min - tenth, y_max + 2*tenth])

    logger.info('Creating cumulative flow graph...')
    ax.figure.savefig('fig.png')
    logger.info('-> graph written to `fig.png`')


def process_flow_category_data(data, since='', until=''):
    """ process cumulative flow status categories from changelog data """

    if data.empty:
        logger.warning('Data for flow analysis is empty')
        return

    if not since or not until:
        raise AnalysisException('Flow analysis requires both `since` and `until` dates for processing')

    dates = pandas.date_range(start=since, end=until, closed='left', freq='D')

    flow_data = data.copy().reset_index()
    flow_data = flow_data.sort_values(['status_change_date'])

    statuses = set(flow_data['status_from_category_name']) | set(flow_data['status_to_category_name'])
    if numpy.nan in statuses:
        statuses.remove(numpy.nan)

    f = pandas.DataFrame(columns=['Date'] + list(statuses))

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
        for item in date_changes['status_from_category_name']:
            if counter[item] > 0:
                counter[item] -= 1
        for item in date_changes['status_to_category_name']:
            counter[item] += 1

        row = dict(counter)
        row['Date'] = date
        f = f.append(row, ignore_index=True)

        last_counter = counter

    f = f.fillna(0)
    if f.empty:
        return f

    f['Date'] = f['Date'].dt.normalize()
    f['Date'] = f['Date'].dt.date

    f = f.set_index('Date')

    return f


def process_issue_data(data, since='', until='', exclude_weekends=False):
    """
    process aggregate issue data from changelog data, calculating cycle and lead times

    optionally, exclude weekends in the cycle time and lead time calculations

    """
    if data.empty:
        logger.warning('Data for issue analysis is empty')
        return

    # filter out issues before since date and after until
    if since:
        data = data[data['issue_created_date'] >= pandas.to_datetime(since)]

    if until:
        data = data[data['issue_created_date'] < pandas.to_datetime(until)]

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

    # What we want to be able to do at this point is to know the total time an
    # issue spends in the "In Progress" state. We could take a look at all of
    # the state changes and compute the sum of the time residing in the
    # "In Progess" state. An alternative that is easier to compute
    # (with less accuracy) is to track when the issue was first created,
    # when it first moved into work in progress, and when it finally
    # completed, ignoring other state transitions in between.
    #
    # To do this without having to map each status, we use the Jira field
    # "status_category_name", which is an ENUM:
    #
    # * To Do
    # * In Progress
    # * Done

    issue_statuses = collections.defaultdict(dict)

    for issue_id, issue in issues.items():
        for update in issue:
            # skip changelogs that are none or nan
            if update.changelog_id is None or numpy.isnan(update.changelog_id):
                continue

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

            issue_statuses[issue_id]['prev_update'] = issue_statuses.get(issue_id, {}).get('last_update', {})
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

        # compute cycle time
        lead_time = pandas.Timedelta(days=0)
        cycle_time = pandas.Timedelta(days=0)

        if complete:
            lead_time = complete - new
            cycle_time = lead_time

            if in_progress:
                cycle_time = complete - in_progress
            else:
                cycle_time = pandas.Timedelta(days=0)

        # adjust lead time and cycle time for weekend (non-working) days
        if complete and exclude_weekends:
            weekend_days = numpy.busday_count(new.date(), complete.date(), weekmask='Sat Sun')
            lead_time -= pandas.Timedelta(days=weekend_days)

            if in_progress:
                weekend_days = numpy.busday_count(in_progress.date(), complete.date(), weekmask='Sat Sun')
                cycle_time -= pandas.Timedelta(days=weekend_days)

        # ensure we don't have negative lead times / cycle times
        if lead_time/pandas.to_timedelta(1, unit='D') < 0:
            lead_time = pandas.Timedelta(days=0)

        if cycle_time/pandas.to_timedelta(1, unit='D') < 0:
            cycle_time = pandas.Timedelta(days=0)

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

    # truncate days
    issue_data['new_day'] = issue_data['new'].values.astype('<M8[D]')

    issue_data['in_progress_day'] = issue_data['in_progress'].values.astype('<M8[D]')

    issue_data['complete_day'] = issue_data['complete'].values.astype('<M8[D]')

    # add column for lead time represented as days
    issue_data['lead_time_days'] = issue_data['lead_time'] / pandas.to_timedelta(1, unit='D')

    # round lead time less than 1 hour to zero
    issue_data.loc[issue_data['lead_time_days'] < 1/24.0, 'lead_time_days'] = 0

    # add column for cycle time represented as days
    issue_data['cycle_time_days'] = issue_data['cycle_time'] / pandas.to_timedelta(1, unit='D')

    # round cycle time less than 1 hour to zero
    issue_data.loc[issue_data['cycle_time_days'] < 1/24.0, 'cycle_time_days'] = 0

    # add column for the last statuses of this issue
    issue_data['prev_issue_status'] = [issue_statuses[issue_ids[key]].get('prev_update', {}).get('status_to_name') for key in issue_data['issue_key']]
    issue_data['prev_issue_status_change_date'] = [issue_statuses[issue_ids[key]].get('prev_update', {}).get('status_change_date') for key in issue_data['issue_key']]
    issue_data['prev_issue_status_category'] = [issue_statuses[issue_ids[key]].get('prev_update', {}).get('status_to_category_name') for key in issue_data['issue_key']]

    # add column for the last statuses of this issue
    issue_data['last_issue_status'] = [issue_statuses[issue_ids[key]].get('last_update', {}).get('status_to_name') for key in issue_data['issue_key']]
    issue_data['last_issue_status_change_date'] = [issue_statuses[issue_ids[key]].get('last_update', {}).get('status_change_date') for key in issue_data['issue_key']]
    issue_data['last_issue_status_category'] = [issue_statuses[issue_ids[key]].get('last_update', {}).get('status_to_category_name') for key in issue_data['issue_key']]

    # set the index
    issue_data = issue_data.set_index('issue_key')

    extra = (
        issues,
        issue_ids,
        issue_keys,
        issue_types,
        issue_points,
        issue_statuses,
    )

    return issue_data, extra


def process_lead_data(issue_data, since='', until=''):
    """
    process lead time data from issue data

    """
    if issue_data.empty:
        logger.warning('Data for lead time analysis is empty')
        return

    lead_data = issue_data.copy()
    lead_data = lead_data.sort_values(['complete'])

    if since:
        lead_data = lead_data[lead_data['complete_day'] >= pandas.to_datetime(since)]

    if until:
        lead_data = lead_data[lead_data['complete_day'] < pandas.to_datetime(until)]

    # drop issues with a lead time less than 1 hour
    lead_data = lead_data[lead_data['lead_time_days'] > (1/24.0)]

    data = pandas.DataFrame()
    data['Create Date'] = lead_data['new_day']
    data['Complete Date'] = lead_data['complete_day']
    data['Lead Time'] = lead_data['lead_time_days']
    data['Moving Average (10 items)'] = lead_data['lead_time_days'].rolling(window=10).mean()
    data['Moving Standard Deviation (10 items)'] = lead_data['lead_time_days'].rolling(window=10).std()
    data['Average'] = lead_data['lead_time_days'].mean()
    data['Standard Deviation'] = lead_data['lead_time_days'].std()
    data = data.rename_axis('Work Item')

    return data


def process_cycle_data(issue_data, since='', until=''):
    """
    process cycle time data from issue data

    """
    if issue_data.empty:
        logger.warning('Data for cycle analysis is empty')
        return

    cycle_data = issue_data.copy()
    cycle_data = cycle_data.sort_values(['complete'])

    if since:
        cycle_data = cycle_data[cycle_data['complete_day'] >= pandas.to_datetime(since)]

    if until:
        cycle_data = cycle_data[cycle_data['complete_day'] < pandas.to_datetime(until)]

    # drop issues with a cycle time less than 1 hour
    cycle_data = cycle_data[cycle_data['cycle_time_days'] > (1/24.0)]

    data = pandas.DataFrame()
    data['In Progress Date'] = cycle_data['in_progress_day']
    data['Complete Date'] = cycle_data['complete_day']
    data['Cycle Time'] = cycle_data['cycle_time_days']
    data['Moving Average (10 items)'] = cycle_data['cycle_time_days'].rolling(window=10).mean()
    data['Moving Standard Deviation (10 items)'] = cycle_data['cycle_time_days'].rolling(window=10).std()
    data['Average'] = cycle_data['cycle_time_days'].mean()
    data['Standard Deviation'] = cycle_data['cycle_time_days'].std()
    data = data.rename_axis('Work Item')

    return data


def process_throughput_data(issue_data, since='', until=''):
    """
    process throughput data from issue data

    """
    if issue_data.empty:
        logger.warning('Data for throughput analysis is empty')
        return

    throughput_data = issue_data.copy()
    throughput_data = throughput_data.sort_values(['complete'])

    if since:
        throughput_data = throughput_data[throughput_data['complete_day'] >= pandas.to_datetime(since)]

    if until:
        throughput_data = throughput_data[throughput_data['complete_day'] < pandas.to_datetime(until)]

    points_data = pandas.pivot_table(throughput_data, values='issue_points', index='complete_day', aggfunc=numpy.sum)

    throughput = pandas.crosstab(throughput_data.complete_day, issue_data.issue_type, colnames=[None]).reset_index()

    date_range = pandas.date_range(start=since, end=until, closed='left', freq='D')

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
    )

    throughput_per_week['Moving Average (4 weeks)'] = throughput_per_week['Throughput'].rolling(window=4).mean()
    throughput_per_week['Moving Standard Deviation (4 weeks)'] = throughput_per_week['Throughput'].rolling(window=4).std()
    throughput_per_week['Average'] = throughput_per_week['Throughput'].mean()
    throughput_per_week['Standard Deviation'] = throughput_per_week['Throughput'].std()

    return throughput, throughput_per_week


def process_wip_data(issue_data, since='', until=''):
    """
    process wip average data from issue data

    """
    if issue_data.empty:
        logger.warning('Data for wip analysis is empty')
        return

    wip_data = issue_data[issue_data['in_progress_day'].notnull()]
    wip_data = wip_data[wip_data['last_issue_status_category'] != 'To Do']
    wip_data = wip_data.sort_values(['in_progress'])

    date_range = pandas.date_range(start=since, end=until, closed='left', freq='D')

    wip = pandas.DataFrame(columns=['Date', 'Work In Progress'])

    for date in date_range:
        date_changes = wip_data
        date_changes = date_changes[date_changes['in_progress_day'] <= date]
        date_changes = date_changes[(date_changes['complete_day'].isnull()) | (date_changes['complete_day'] > date)]

        row = dict()
        row['Date'] = date
        row['Work In Progress'] = len(date_changes)
        wip = wip.append(row, ignore_index=True)

    wip = wip.set_index('Date')
    wip = wip.reindex(date_range).fillna(0).astype(int).rename_axis('Date')

    wip['Moving Average (10 days)'] = wip['Work In Progress'].rolling(window=10).mean()
    wip['Moving Standard Deviation (10 days)'] = wip['Work In Progress'].rolling(window=10).std()
    wip['Average'] = wip['Work In Progress'].mean()
    wip['Standard Deviation'] = wip['Work In Progress'].std()

    # resample to also provide how much wip we have at the end of each week
    wip_per_week = pandas.DataFrame(
        wip['Work In Progress'].resample('W-Mon').last()
    )

    wip_per_week['Moving Average (4 weeks)'] = wip_per_week['Work In Progress'].rolling(window=4).mean()
    wip_per_week['Moving Standard Deviation (4 weeks)'] = wip_per_week['Work In Progress'].rolling(window=4).std()
    wip_per_week['Average'] = wip_per_week['Work In Progress'].mean()
    wip_per_week['Standard Deviation'] = wip_per_week['Work In Progress'].std()

    return wip, wip_per_week


def process_wip_age_data(issue_data, since='', until=''):
    """
    process wip age data from issue data

    """
    if issue_data.empty:
        logger.warning('Data for wip age analysis is empty')
        return

    age_data = issue_data[issue_data['in_progress_day'].notnull()]

    if since:
        age_data = age_data[age_data['in_progress_day'] >= pandas.to_datetime(since)]

    if until:
        age_data = age_data[age_data['in_progress_day'] < pandas.to_datetime(until)]

    # only compute ages for incomplete work, i.e., the completion day has not occurred or is outside the filter range
    age_data = age_data[(age_data['complete_day'].isnull()) | (age_data['complete_day'] >= pandas.to_datetime(until))]
    age_data = age_data[age_data['last_issue_status_category'] != 'To Do']
    age_data = age_data.sort_values(['in_progress'])

    today = pandas.to_datetime(until)

    age_data['First In Progress'] = age_data['in_progress_day']
    age_data['Stage'] = age_data['last_issue_status']
    age_data['Age in Stage'] = (today - age_data['last_issue_status_change_date']) / pandas.to_timedelta(1, unit='D')
    age_data['Age'] = (today - age_data['in_progress']) / pandas.to_timedelta(1, unit='D')
    age_data['Average'] = age_data['Age'].mean()
    age_data['Standard Deviation'] = age_data['Age'].std()
    age_data['P50'] = age_data['Age'].quantile(0.5)
    age_data['P75'] = age_data['Age'].quantile(0.75)
    age_data['P85'] = age_data['Age'].quantile(0.85)
    age_data['P95'] = age_data['Age'].quantile(0.95)
    age_data['P99'] = age_data['Age'].quantile(0.999)

    # fix negative age in stages (because of an until that is set before completion date)
    age_data.loc[age_data['Age in Stage'] < 0, 'Stage'] = 'Unknown'
    age_data.loc[age_data['Age in Stage'] < 0, 'Age in Stage'] = age_data.loc[age_data['Age in Stage'] < 0, 'Age']

    return age_data


def process_correlation(x, y):
    """
    run a pearson correlation analysis between two sets (usually issue_points and cycle_time_days)

    """
    return pingouin.corr(x, y, method='pearson')


def analyze_survival_km(issue_data, since='', until=''):
    """
    run a kaplan-meier survivability analysis on the issue data

    """
    survivability_data = issue_data.copy()
    survivability_data = survivability_data[survivability_data['complete_day'] >= pandas.to_datetime(since)]
    survivability_data = survivability_data[survivability_data['complete_day'] < pandas.to_datetime(until)]
    survivability_data = survivability_data.sort_values(['complete_day'])

    durations = survivability_data['cycle_time_days']
    event_observed = [1 if c else 0 for c in survivability_data['cycle_time_days']]

    km = lifelines.KaplanMeierFitter()

    return km.fit(durations, event_observed, label='Kaplan Meier Estimate'), km


def analyze_survival_wb(issue_data, since='', until=''):
    """
    run a weibull survivability analysis on the issue data

    """
    survivability_data = issue_data.copy()
    survivability_data = survivability_data[survivability_data['complete_day'] >= pandas.to_datetime(since)]
    survivability_data = survivability_data[survivability_data['complete_day'] < pandas.to_datetime(until)]
    survivability_data = survivability_data.sort_values(['complete_day'])

    durations = [c if c else 0.00001 for c in survivability_data['cycle_time_days']]
    event_observed = [1 if c else 0 for c in survivability_data['cycle_time_days']]

    wb = lifelines.WeibullFitter()

    return wb.fit(durations, event_observed, label='Weibull Estimate'), wb


def forecast_montecarlo_how_long_items(throughput_data, items=10, simulations=10000, window=90):
    """
    forecast number of days it will take to complete n number of items

    """
    if throughput_data.empty:
        logger.warning('Data for Montecarlo analysis is empty')
        return

    SIMULATION_ITEMS = items
    SIMULATIONS = simulations
    LAST_DAYS = window

    logger.info('Running Montecarlo analysis...')

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
        logger.warning(f'Montecarlo window ({window}) is larger than throughput dataset ({count}). '
                       f'Try increasing your date filter to include more observations or decreasing the forecast window size.')

    samples = []
    for i in range(SIMULATIONS):
        if (i+1) % 1000 == 0:
            logger.info(f'-> {i+1} simulations run')
        samples.append(simulate_days(dataset, SIMULATION_ITEMS))
    logger.info('---')
    samples = pandas.DataFrame(samples, columns=['Days'])
    distribution_how_long = samples.groupby(['Days']).size().reset_index(name='Frequency')
    distribution_how_long = distribution_how_long.sort_index(ascending=False)
    distribution_how_long['Probability'] = 100 - 100 * distribution_how_long.Frequency.cumsum()/distribution_how_long.Frequency.sum()

    return distribution_how_long, samples


def forecast_montecarlo_how_many_items(throughput_data, days=10, simulations=10000, window=90):
    """
    forecast number of items to be completed in n days

    """
    if throughput_data.empty:
        logger.warning('Data for Montecarlo analysis is empty')
        return

    SIMULATION_DAYS = days
    SIMULATIONS = simulations
    LAST_DAYS = window

    logger.info('Running Montecarlo analysis...')

    dataset = throughput_data[['Throughput']].tail(LAST_DAYS).reset_index(drop=True)
    count = len(dataset)
    if count < window:
        logger.warning(f'Montecarlo window ({window}) is larger than throughput dataset ({count}). '
                       f'Try increasing your date filter to include more observations or decreasing the forecast window size.')

    samples = []
    for i in range(SIMULATIONS):
        if (i+1) % 1000 == 0:
            logger.info(f'-> {i+1} simulations run')
        samples.append(dataset.sample(n=SIMULATION_DAYS, replace=True).sum()['Throughput'])
    logger.info('---')
    samples = pandas.DataFrame(samples, columns=['Items'])
    distribution_how = samples.groupby(['Items']).size().reset_index(name='Frequency')
    distribution_how = distribution_how.sort_index(ascending=False)
    distribution_how['Probability'] = 100 * distribution_how.Frequency.cumsum()/distribution_how.Frequency.sum()

    return distribution_how, samples


def forecast_montecarlo_how_long_points(throughput_data, points=10, simulations=10000, window=90):
    """
    forecast number of days it will take to complete n number of points

    """
    if throughput_data.empty:
        logger.warning('Data for Montecarlo analysis is empty')
        return

    SIMULATION_ITEMS = points
    SIMULATIONS = simulations
    LAST_DAYS = window

    logger.info('Running Montecarlo analysis...')

    if (throughput_data['Velocity']/throughput_data['Throughput']).max() == 1:
        logger.warning('All velocity data is equal. Did you load data with points fields?')

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
        logger.warning(f'Montecarlo window ({window}) is larger than velocity dataset ({count}). '
                       f'Try increasing your date filter to include more observations or decreasing the forecast window size.')

    samples = []
    for i in range(SIMULATIONS):
        if (i+1) % 1000 == 0:
            logger.info(f'-> {i+1} simulations run')
        samples.append(simulate_days(dataset, SIMULATION_ITEMS))
    logger.info('---')
    samples = pandas.DataFrame(samples, columns=['Days'])
    distribution_how_long = samples.groupby(['Days']).size().reset_index(name='Frequency')
    distribution_how_long = distribution_how_long.sort_index(ascending=False)
    distribution_how_long['Probability'] = 100 - 100 * distribution_how_long.Frequency.cumsum()/distribution_how_long.Frequency.sum()

    return distribution_how_long, samples


def forecast_montecarlo_how_many_points(throughput_data, days=10, simulations=10000, window=90):
    """
    forecast number of points to be completed in n days

    """
    if throughput_data.empty:
        logger.warning('Data for Montecarlo analysis is empty')
        return

    SIMULATION_DAYS = days
    SIMULATIONS = simulations
    LAST_DAYS = window

    logger.info('Running Montecarlo analysis...')

    if (throughput_data['Velocity']/throughput_data['Throughput']).max() == 1:
        logger.warning('All velocity data is equal. Did you load data with points fields?')

    dataset = throughput_data[['Velocity']].tail(LAST_DAYS).reset_index(drop=True)
    count = len(dataset)
    if count < window:
        logger.warning(f'Montecarlo window ({window}) is larger than velocity dataset ({count}). '
                       f'Try increasing your date filter to include more observations or decreasing the forecast window size.')

    samples = []
    for i in range(SIMULATIONS):
        if (i+1) % 1000 == 0:
            logger.info(f'-> {i+1} simulations run')
        samples.append(dataset.sample(n=SIMULATION_DAYS, replace=True).sum()['Velocity'])
    logger.info('---')
    samples = pandas.DataFrame(samples, columns=['Points'])
    distribution_how = samples.groupby(['Points']).size().reset_index(name='Frequency')
    distribution_how = distribution_how.sort_index(ascending=False)
    distribution_how['Probability'] = 100 * distribution_how.Frequency.cumsum()/distribution_how.Frequency.sum()

    return distribution_how, samples


def output_formatted_data(output, title, data, output_exclude_title=False, output_header='', output_footer='', output_columns=None, output_format=None):
    """
    format the output of a data frame with a header, footer, and formatting options

    """
    if output_format is None:
        output_format = 'string'

    output.writelines(f'{line}\n' for line in [
        output_header if output_header else '',
        f'# {title}' if title and not output_exclude_title else '',
        data.to_string(columns=output_columns) if output_format == 'string' else
        data.to_csv(columns=output_columns) if output_format == 'csv' else
        data.to_html(columns=output_columns) if output_format == 'html' else '',
        output_footer if output_footer else '',
    ] if line)


def cmd_summary(output, issue_data, since='', until=''):
    """ process summary command """
    # current lead time
    lt = process_lead_data(issue_data, since=since, until=until)

    # current cycle time
    c = process_cycle_data(issue_data, since=since, until=until)

    # current throughput
    t, tw = process_throughput_data(issue_data, since=since, until=until)

    # current wip
    w, ww = process_wip_data(issue_data, since=since, until=until)
    a = process_wip_age_data(issue_data, since=since, until=until)

    lead_time = pandas.DataFrame.from_records([
        ('Average', lt['Average'].iat[-1]),
        ('Standard Deviation', lt['Standard Deviation'].iat[-1]),
        ('Moving Average (10 items)', lt['Moving Average (10 items)'].iat[-1]),
        ('Moving Standard Deviation (10 items)', lt['Moving Standard Deviation (10 items)'].iat[-1]),
    ], columns=('Metric', 'Value'), index='Metric')

    cycle_time = pandas.DataFrame.from_records([
        ('Average', c['Average'].iat[-1]),
        ('Standard Deviation', c['Standard Deviation'].iat[-1]),
        ('Moving Average (10 items)', c['Moving Average (10 items)'].iat[-1]),
        ('Moving Standard Deviation (10 items)', c['Moving Standard Deviation (10 items)'].iat[-1]),
    ], columns=('Metric', 'Value'), index='Metric')

    throughput = pandas.DataFrame.from_records([
        ('Average', t['Average'].iat[-1]),
        ('Standard Deviation', t['Standard Deviation'].iat[-1]),
        ('Moving Average (10 days)', t['Moving Average (10 days)'].iat[-1]),
        ('Moving Standard Deviation (10 days)', t['Moving Standard Deviation (10 days)'].iat[-1]),
    ], columns=('Metric', 'Value'), index='Metric')

    throughput_weekly = pandas.DataFrame.from_records([
        ('Average', tw['Average'].iat[-1]),
        ('Standard Deviation', tw['Standard Deviation'].iat[-1]),
        ('Moving Average (4 weeks)', tw['Moving Average (4 weeks)'].iat[-1]),
        ('Moving Standard Deviation (4 weeks)', tw['Moving Standard Deviation (4 weeks)'].iat[-1]),
    ], columns=('Metric', 'Value'), index='Metric')

    wip = pandas.DataFrame.from_records([
        ('Average', w['Average'].iat[-1]),
        ('Standard Deviation', w['Standard Deviation'].iat[-1]),
        ('Moving Average (10 days)', w['Moving Average (10 days)'].iat[-1]),
        ('Moving Standard Deviation (10 days)', w['Moving Standard Deviation (10 days)'].iat[-1]),
    ], columns=('Metric', 'Value'), index='Metric')

    wip_weekly = pandas.DataFrame.from_records([
        ('Average', ww['Average'].iat[-1]),
        ('Standard Deviation', ww['Standard Deviation'].iat[-1]),
        ('Moving Average (4 weeks)', ww['Moving Average (4 weeks)'].iat[-1]),
        ('Moving Standard Deviation (4 weeks)', ww['Moving Standard Deviation (4 weeks)'].iat[-1]),
    ], columns=('Metric', 'Value'), index='Metric')

    wip_age = pandas.DataFrame.from_records([
        ('Average', a['Average'].iat[-1]),
        ('50th Percentile', a['P50'].iat[-1]),
        ('75th Percentile', a['P75'].iat[-1]),
        ('85th Percentile', a['P85'].iat[-1]),
        ('95th Percentile', a['P95'].iat[-1]),
    ], columns=('Metric', 'Value'), index='Metric')

    output_formatted_data(output, 'Lead Time', lead_time)
    output_formatted_data(output, 'Cycle Time', cycle_time)
    output_formatted_data(output, 'Throughput (Daily)', throughput)
    output_formatted_data(output, 'Throughput (Weekly)', throughput_weekly)
    output_formatted_data(output, 'Work In Progress (Daily)', wip)
    output_formatted_data(output, 'Work In Progress (Weekly)', wip_weekly)
    output_formatted_data(output, f'Work In Progress Age (ending {until})', wip_age)


def cmd_correlation(output, issue_data, since='', until=''):
    """ process correlation command """
    points = issue_data['issue_points'].values.astype(float)
    lead_time = issue_data['lead_time_days'].values.astype(float)
    cycle_time = issue_data['cycle_time_days'].values.astype(float)
    cycle_result = process_correlation(points, cycle_time)
    lead_result = process_correlation(points, lead_time)

    point_summary = pandas.DataFrame.from_records([
       ('Observations', issue_data['issue_points'].count()),
       ('Min', issue_data['issue_points'].min()),
       ('Max', issue_data['issue_points'].max()),
       ('Average', issue_data['issue_points'].mean()),
       ('Standard Deviation', issue_data['issue_points'].std()),
    ], columns=('Metric', 'Value'), index='Metric')

    cycle_correlation_summary = pandas.DataFrame.from_records([
        ('Observations (n)', cycle_result['n'].iat[0]),
        ('Correlation Coefficient (r)', cycle_result['r'].iat[0]),
        ('Determination Coefficient (r^2)', cycle_result['r2'].iat[0]),
        ('P-Value (p)', cycle_result['p-val'].iat[0]),
        ('Likelihood of detecting effect (power)', cycle_result['power'].iat[0]),
        ('Significance (p <= 0.05)', 'significant' if cycle_result['p-val'].iat[0] <= 0.05 else 'not significant'),
    ], columns=('Metric', 'Value'), index='Metric')

    lead_correlation_summary = pandas.DataFrame.from_records([
        ('Observations (n)', lead_result['n'].iat[0]),
        ('Correlation Coefficient (r)', lead_result['r'].iat[0]),
        ('Determination Coefficient (r^2)', lead_result['r2'].iat[0]),
        ('P-Value (p)', lead_result['p-val'].iat[0]),
        ('Likelihood of detecting effect (power)', lead_result['power'].iat[0]),
        ('Significance (p <= 0.05)', 'significant' if lead_result['p-val'].iat[0] <= 0.05 else 'not significant'),
    ], columns=('Metric', 'Value'), index='Metric')

    output_formatted_data(output, 'Points', point_summary)
    output_formatted_data(output, 'Points', point_summary)
    output_formatted_data(output, 'Point Correlation to Cycle Time', cycle_correlation_summary)
    output_formatted_data(output, 'Point Correlation to Lead Time', lead_correlation_summary)


def cmd_detail_flow(output, data, since='', until=''):
    """ process flow command """
    flow_data = process_flow_data(data, since=since, until=until)
    output_formatted_data(output, 'Cumulative Flow', flow_data)


def cmd_detail_wip(output, issue_data, wip_type='', since='', until=''):
    """ process wip command """
    # current wip
    w, ww = process_wip_data(issue_data, since=since, until=until)
    a = process_wip_age_data(issue_data, since=since, until=until)

    if wip_type == 'daily':
        output_formatted_data(output, 'Work In Progress (Daily)', w)

    if wip_type == 'weekly':
        output_formatted_data(output, 'Work In Progress (Weekly)', ww)

    if wip_type == 'aging':
        wa = a[['First In Progress', 'Age', 'Stage', 'Age in Stage']]
        output_formatted_data(output, f'Work In Progress Age (ending {until})', wa)


def cmd_detail_throughput(output, issue_data, since='', until='', throughput_type=''):
    """ process throughput command """
    # current throughput
    t, tw = process_throughput_data(issue_data, since=since, until=until)

    if throughput_type == 'daily':
        output_formatted_data(output, 'Throughput (Daily)', t)

    if throughput_type == 'weekly':
        output_formatted_data(output, 'Throughput (Weekly)', tw)


def cmd_detail_leadtime(output, issue_data, since='', until=''):
    """ process lead time command """
    # current lead time
    c = process_lead_data(issue_data, since=since, until=until)
    output_formatted_data(output, 'Lead Time', c)


def cmd_detail_cycletime(output, issue_data, since='', until=''):
    """ process cycletime command """
    # current cycle time
    c = process_cycle_data(issue_data, since=since, until=until)
    output_formatted_data(output, 'Cycle Time', c)


def cmd_survival_km(output, issue_data, since='', until=''):
    """ process survival analysis using Kaplan-Meier """
    m, _ = analyze_survival_km(issue_data, since=since, until=until)

    km_summary = pandas.DataFrame.from_records([
        (f'{int((q)*100)}%', m.percentile(1-q)) for q in (0.25, 0.5, 0.75, 0.85, 0.95, 0.999)
    ], columns=('Probability', 'Days'), index='Probability')

    output_formatted_data(output, 'Kaplan-Meier Estimation: Within how many days can a single work item be completed?', km_summary)


def cmd_survival_wb(output, issue_data, since='', until=''):
    """ process survival analysis using Weibull """
    m, _ = analyze_survival_wb(issue_data, since=since, until=until)

    wb_summary = pandas.DataFrame.from_records([
        (f'{int((q)*100)}%', m.percentile(1-q)) for q in (0.25, 0.5, 0.75, 0.85, 0.95, 0.999)
    ], columns=('Probability', 'Days'), index='Probability')

    output_formatted_data(output, 'Weibull Estimation: Within how many days can a single work item be completed?', wb_summary)


def cmd_forecast_items_n(output, issue_data, since='', until='', n=10, simulations=10000, window=90):
    """ process forecast items n command """
    # pre-req
    t, tw = process_throughput_data(issue_data, since=since, until=until)

    # analysis
    ml, s = forecast_montecarlo_how_long_items(t, items=n, simulations=simulations, window=window)

    forecast_summary = pandas.DataFrame.from_records([
        (f'{int(q*100)}%', s.Days.quantile(q)) for q in (0.25, 0.5, 0.75, 0.85, 0.95, 0.999)
    ], columns=('Probability', 'Days'), index='Probability')

    output_formatted_data(output, f'Montecarlo Forecast: Within how many days can {n} work items be completed?', forecast_summary)


def cmd_forecast_items_days(output, issue_data, since='', until='', days=10, simulations=10000, window=90):
    """ process forecast items days command """
    # pre-req
    t, tw = process_throughput_data(issue_data, since=since, until=until)

    # analysis
    mh, s = forecast_montecarlo_how_many_items(t, days=days, simulations=simulations, window=window)

    forecast_summary = pandas.DataFrame.from_records([
        (f'{int(q*100)}%', s.Items.quantile(1-q)) for q in (0.25, 0.5, 0.75, 0.85, 0.95, 0.999)
    ], columns=('Probability', 'Items'), index='Probability')

    output_formatted_data(output, f'Montecarlo Forecast: How many work items can be completed within {days} days?', forecast_summary)


def cmd_forecast_points_n(output, issue_data, since='', until='', n=10, simulations=10000, window=90):
    """ process forecast points n command """
    # pre-req
    t, tw = process_throughput_data(issue_data, since=since, until=until)

    # analysis
    ml, s = forecast_montecarlo_how_long_points(t, points=n, simulations=simulations, window=window)

    forecast_summary = pandas.DataFrame.from_records([
        (f'{int(q*100)}%', s.Days.quantile(q)) for q in (0.25, 0.5, 0.75, 0.85, 0.95, 0.999)
    ], columns=('Probability', 'Days'), index='Probability')

    output_formatted_data(output, f'Montecarlo Forecast: Within how many days can {n} points be completed?', forecast_summary)


def cmd_forecast_points_days(output, issue_data, since='', until='', days=10, simulations=10000, window=90):
    """ process forecast points days command """
    # pre-req
    t, tw = process_throughput_data(issue_data, since=since, until=until)

    # analysis
    mh, s = forecast_montecarlo_how_many_points(t, days=days, simulations=simulations, window=window)

    forecast_summary = pandas.DataFrame.from_records([
        (f'{int(q*100)}%', s.Points.quantile(1-q)) for q in (0.25, 0.5, 0.75, 0.85, 0.95, 0.999)
    ], columns=('Probability', 'points'), index='Probability')

    output_formatted_data(output, f'Montecarlo Forecast: How many points can be completed within {days} days?', forecast_summary)


def cmd_shell(output, issue_data, since='', until='', args=None):
    logger.info('Creating interactive Python shell...')
    logger.info('-> locals: %s' % ', '.join(locals().keys()))
    logger.info('---')
    code.interact(local=locals())


def run(args):
    """
    run the utility in response to the command line arguments

    """
    data, dupes, filtered = read_data(args.file, exclude_types=args.exclude_type, since=args.since, until=args.until)
    if data.empty:
        logger.warning('Data for analysis is empty')
        return

    exclude_weekends = args.exclude_weekends

    # if no since/until is provided, compute the range from the data
    since = args.since
    if not since:
        since = min(data['issue_created_date'].min(), data['status_change_date'].min())
    until = args.until
    if not until:
        until = max(data['issue_created_date'].max(), data['status_change_date'].max()) + pandas.Timedelta(1, 'D')

    output = args.output

    # preprocess issue data
    i, _ = process_issue_data(data, since=since, until=until, exclude_weekends=exclude_weekends)

    # summary
    if args.command == 'summary':
        cmd_summary(output, i, since=since, until=until)

    # detail
    if args.command == 'detail' and args.detail_type == 'flow':
        cmd_detail_flow(output, data, since=since, until=until)

    if args.command == 'detail' and args.detail_type == 'wip':
        cmd_detail_wip(output, i, since=since, until=until, wip_type=args.type)

    if args.command == 'detail' and args.detail_type == 'throughput':
        cmd_detail_throughput(output, i, since=since, until=until, throughput_type=args.type)

    if args.command == 'detail' and args.detail_type == 'cycletime':
        cmd_detail_cycletime(output, i, since=since, until=until)

    if args.command == 'detail' and args.detail_type == 'leadtime':
        cmd_detail_leadtime(output, i, since=since, until=until)

    # correlation
    if args.command == 'correlation':
        cmd_correlation(output, i, since=since, until=until)

    # survival
    if args.command == 'survival' and args.survival_type == 'km':
        cmd_survival_km(output, i, since=since, until=until)

    if args.command == 'survival' and args.survival_type == 'wb':
        cmd_survival_wb(output, i, since=since, until=until)

    # forecast
    if args.command == 'forecast' and args.forecast_type == 'items' and args.n:
        cmd_forecast_items_n(output, i, since=since, until=until, n=args.n, simulations=args.simulations, window=args.window)

    if args.command == 'forecast' and args.forecast_type == 'items' and args.days:
        cmd_forecast_items_days(output, i, since=since, until=until, days=args.days, simulations=args.simulations, window=args.window)

    if args.command == 'forecast' and args.forecast_type == 'points' and args.n:
        cmd_forecast_points_n(output, i, since=since, until=until, n=args.n, simulations=args.simulations, window=args.window)

    if args.command == 'forecast' and args.forecast_type == 'points' and args.days:
        cmd_forecast_points_days(output, i, since=since, until=until, days=args.days, simulations=args.simulations, window=args.window)

    # shell
    if args.command == 'shell':
        cmd_shell(output, i, since=since, until=until, args=args)


def main():
    """
    parse the command line arguments and run the utility

    """
    global output_formatted_data

    parser = argparse.ArgumentParser(description='Analyze changelog data')
    parser.add_argument('-f', '--file', type=argparse.FileType('r'), default='-', help='Data file to analyze (default: stdin)')
    parser.add_argument('-o', '--output', type=argparse.FileType('w'), default='-', help='File to output results (default: stdout)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Be quiet and only log warnings to console.')

    parser.add_argument('--exclude-weekends', action='store_true', help='Exclude weekends from cycle and lead time calculations')
    parser.add_argument('--exclude-type', metavar='TYPE', action='append', help='Exclude one or more specific types from the analysis (e.g., "Epic", "Bug", etc)')
    parser.add_argument('--since', help='Only process work items created since date (format: YYYY-MM-DD)')
    parser.add_argument('--until', help='Only process work items created up until date (format: YYYY-MM-DD)')

    subparsers = parser.add_subparsers(dest='command')

    def add_output_params(subparser):
        subparser.add_argument('--column', dest='output_columns', action='append', help='Filter output to only include this (can accept more than one for ordering')
        subparser.add_argument('--format', dest='output_format', choices=('string', 'csv', 'html'), default='string',
            help='Which output format should be used (string, csv, html)') # NOQA
        subparser.add_argument('--header', dest='output_header', help='Prepend each data table with header text')
        subparser.add_argument('--footer', dest='output_footer', default=' ', help='Append each data table with footer text (default: \\n')
        subparser.add_argument('--exclude-title', dest='output_exclude_title', action='store_true', help='Exclude title of data table in output')

    # summary
    subparser_summary = subparsers.add_parser('summary', help='Generate a summary of metric data (cycle time, throughput, wip)')  # NOQA
    add_output_params(subparser_summary)

    # detail
    subparser_detail = subparsers.add_parser('detail', help='Output detailed analysis data')
    subparser_detail_subparsers = subparser_detail.add_subparsers(dest='detail_type')

    subparser_flow = subparser_detail_subparsers.add_parser('flow', help='Analyze cumulative flow and output detail')
    add_output_params(subparser_flow)

    subparser_wip = subparser_detail_subparsers.add_parser('wip', help='Analyze wip and output detail')
    subparser_wip.add_argument('type', choices=('daily', 'weekly', 'aging'), help='Type of wip data to output (daily, weekly, aging)')
    add_output_params(subparser_wip)

    subparser_throughput = subparser_detail_subparsers.add_parser('throughput', help='Analyze throughput and output detail')
    subparser_throughput.add_argument('type', choices=('daily', 'weekly'), help='Type of throughput data to output (daily, weekly)')
    add_output_params(subparser_throughput)

    subparser_cycletime = subparser_detail_subparsers.add_parser('cycletime', help='Analyze cycletime and output detail')  # NOQA
    add_output_params(subparser_cycletime)

    subparser_leadtime = subparser_detail_subparsers.add_parser('leadtime', help='Analyze leadtime and output detail')  # NOQA
    add_output_params(subparser_leadtime)

    # correlation
    subparser_corrrelation = subparsers.add_parser('correlation', help='Test correlation between issue_points and lead/cycle times')  # NOQA
    add_output_params(subparser_corrrelation)

    # survival
    subparser_survival = subparsers.add_parser('survival', help='Analyze the survival of work items')
    subparser_survival_subparsers = subparser_survival.add_subparsers(dest='survival_type')

    subparser_survival_km = subparser_survival_subparsers.add_parser('km', help='Analyze the survival of work items using Kaplan-Meier Estimation')  # NOQA
    add_output_params(subparser_survival_km)

    subparser_survival_wb = subparser_survival_subparsers.add_parser('wb', help='Analyze the survival of work items using Weibull Estimation')  # NOQA
    add_output_params(subparser_survival_wb)

    # forecast
    subparser_forecast = subparsers.add_parser('forecast', help='Forecast the future using Montecarlo simulation')
    subparser_forecast_subparsers = subparser_forecast.add_subparsers(dest='forecast_type')

    subparser_forecast_items = subparser_forecast_subparsers.add_parser('items', help='Forecast future work items')
    subparser_forecast_items_group = subparser_forecast_items.add_mutually_exclusive_group(required=True)
    subparser_forecast_items_group.add_argument('-n', '--items',  dest='n', type=int,
        help='Number of items to predict answering the question "within how many days can N items be completed?"')  # NOQA
    subparser_forecast_items_group.add_argument('-d', '--days', type=int, help='Number of days to predict answering the question "how many items can be completed within N days?"')
    subparser_forecast_items.add_argument('--simulations', default=10000, help='Number of simulation iterations to run (default: 10000)')
    subparser_forecast_items.add_argument('--window', default=90, help='Window of historical data to use in the forecast (default: 90 days)')
    add_output_params(subparser_forecast_items)

    subparser_forecast_points = subparser_forecast_subparsers.add_parser('points', help='Forecast future points')
    subparser_forecast_points_group = subparser_forecast_points.add_mutually_exclusive_group(required=True)
    subparser_forecast_points_group.add_argument('-n', '--points', dest='n', type=int,
        help='Number of points to predict answering the question "within how many days can N points be completed?"')  # NOQA
    subparser_forecast_points_group.add_argument('-d', '--days', type=int,
        help='Number of days to predict answering the question "how many points can be completed within N days?"')  # NOQA
    subparser_forecast_points.add_argument('--simulations', default=10000, help='Number of simulation iterations to run (default: 10000)')
    subparser_forecast_points.add_argument('--window', default=90, help='Window of historical data to use in the forecast (default: 90 days)')
    add_output_params(subparser_forecast_points)

    # shell
    subparser_shell = subparsers.add_parser('shell', help='Load the data into an interactive Python shell')  # NOQA

    args = parser.parse_args()

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN if args.quiet else logging.INFO)
    ch.setFormatter(Formatter())
    logger.addHandler(ch)
    logger.setLevel(logging.WARN if args.quiet else logging.INFO)

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'detail' and args.detail_type is None:
        subparser_detail.print_help()
        return

    if args.command == 'survival' and args.survival_type is None:
        subparser_survival.print_help()
        return

    if args.command == 'forecast' and args.forecast_type is None:
        subparser_forecast.print_help()
        return

    # prepare output formatting
    if args.output:
        args.output.reconfigure(line_buffering=True)

    format_args = ['output_exclude_title', 'output_header', 'output_footer', 'output_format', 'output_columns']
    if any(hasattr(args, arg) for arg in format_args):
        kw = {key: getattr(args, key) for key in format_args if hasattr(args, key)}
        output_formatted_data = functools.partial(output_formatted_data, **kw)

    try:
        init()
        run(args)
    except AnalysisException as e:
        logger.error('Error: %s', e)


if __name__ == '__main__':
    main()
