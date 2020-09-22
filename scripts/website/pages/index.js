import Link from 'next/link'
import { Header } from 'semantic-ui-react'
import { List } from 'semantic-ui-react'
import styled from 'styled-components'
import moment from 'moment'

import { Page } from 'comps/page'
import { fileToDate } from 'utils'

/*
reports = {
    "foo": {
      "title": "Foo Report",
      "description": "Weekly report for Foo",
      "files": [
        {
          filename: "foo.csv",
          url: "http://google.com",
        },
      ]
    },
}

apps = {
  "covid_19": {
    "malaysia": {
      "11111111-aaaaaaa": {
        "id": "covid_19/malaysia/1111111-aaaaaaa",
        "app": "covid_19",
        "region": "malaysia",
        "timestamp": 1111111,
        "commit": "aaaaaaa",
        "files": [
          {
            path: "data/foo.db",
            url: "http://google.com",
          },
        ]
      }
    }
  }
}
*/
const IGNORE_MODEL_NAMES = ['test']

export async function getStaticProps() {
  const data = await import('../website.json')
  const { reports, apps } = data.default
  const appData = []
  const reportData = []
  for (let name of Object.keys(reports)) {
    let mostRecent = 0
    let numRuns = 0
    for (let file of reports[name].files) {
      numRuns++
      const timestamp = fileToDate(file)
      if (timestamp > mostRecent) {
        mostRecent = timestamp
      }
    }
    reportData.push({
      name: reports[name].title,
      slug: name,
      mostRecent,
      numRuns,
    })
  }
  for (let name of Object.keys(apps)) {
    if (IGNORE_MODEL_NAMES.includes(name)) continue
    if (Object.keys(reports).includes(name)) continue
    let mostRecent = 0
    let numRuns = 0
    for (let regionName of Object.keys(apps[name])) {
      for (let uuid of Object.keys(apps[name][regionName])) {
        numRuns++
        const run = apps[name][regionName][uuid]
        if (run.timestamp > mostRecent) {
          mostRecent = run.timestamp
        }
      }
    }
    appData.push({
      name,
      mostRecent,
      numRuns,
    })
  }
  return { props: { appData, reportData } }
}

const AppPage = ({ appData, reportData }) => {
  return (
    <Page title="Autumn Data">
      <h1>Autumn Results</h1>
      <List relaxed divided size="medium">
        {reportData
          .sort((a, b) => (a.name > b.name ? 1 : -1))
          .map(({ name, slug, mostRecent, numRuns }) => (
            <List.Item key={name}>
              <Header as="h3">
                <Link href="/report/[name]" as={`/report/${slug}`}>
                  <a>{name.replace('_', ' ')} Report</a>
                </Link>
              </Header>
              <List.Description>
                {numRuns} runs - last run was{' '}
                {moment(mostRecent, 'X').fromNow()}
              </List.Description>
            </List.Item>
          ))}

        {appData
          .sort((a, b) => (a.name > b.name ? 1 : -1))
          .map(({ name, mostRecent, numRuns }) => (
            <List.Item key={name}>
              <Header as="h3">
                <Link href="/app/[name]" as={`/app/${name}`}>
                  <a>{name.replace('_', ' ')} App</a>
                </Link>
              </Header>
              <List.Description>
                {numRuns} runs - last run was{' '}
                {moment(mostRecent, 'X').fromNow()}
              </List.Description>
            </List.Item>
          ))}
      </List>
    </Page>
  )
}

export default AppPage
