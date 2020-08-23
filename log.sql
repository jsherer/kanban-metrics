select 
  i.projectid as project_id,
  i.projectkey as project_key,
  c.issueid as issue_id,
  c.issuekey as issue_key,
  i.issuetypeid as issue_type_id,
  i.issuetypename as issue_type_name,
  i.summary as issue_title,
  c.issuecreateddate as issue_created_date,
  c.historyid as changelog_id,
  c.itemfrom as status_from_id,
  c.itemfromstring as status_from_name,
  c.itemto as status_to_id, 
  c.itemtostring as status_to_name,
  
  case
    when c.itemfromstring = 'To Do' then 'To Do'
    when c.itemfromstring = 'Prioritized' then 'To Do'
    when c.itemfromstring = 'In Progress' then 'In Progress'
    when c.itemfromstring = 'Review' then 'In Progress'
    when c.itemfromstring = 'Accepted' then 'In Progress'
    when c.itemfromstring = 'Can''t Fix' then 'Done'
    when c.itemfromstring = 'Deployed' then 'Done'
  end as status_from_category_name,
  
  case
    when c.itemtostring = 'To Do' then 'To Do'
    when c.itemtostring = 'Prioritized' then 'To Do'
    when c.itemtostring = 'In Progress' then 'In Progress'
    when c.itemtostring = 'Review' then 'In Progress'
    when c.itemtostring = 'Accepted' then 'In Progress'
    when c.itemtostring = 'Can''t Fix' then 'Done'
    when c.itemtostring = 'Deployed' then 'Done'
  end as status_to_category_name,
  
  c.created as status_change_date

from
  dim_jira_issue i
  left join dim_jira_changelog c on i.id = c.issueid and c.itemfield = 'status'
where
      project_key = 'CO'
  and issue_created_date >= '2020-01-01'::date
  and status_change_date >= '2020-01-01'::date
order by c.created asc