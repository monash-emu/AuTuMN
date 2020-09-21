import Link from 'next/link'
import { Header } from 'semantic-ui-react'
import { List } from 'semantic-ui-react'
import styled from 'styled-components'
import moment from 'moment'

import { Page } from 'comps/page'

/*
dhhs = [
  {
    filename: "foo.csv",
    url: "http://google.com",
  },
]

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
const IGNORE_MODEL_NAMES = ['test', 'dhhs']

export async function getStaticProps() {
  const data = await import('../website.json')
  const { dhhs, apps } = data.default
  const appData = []
  for (let appName of Object.keys(apps)) {
    if (IGNORE_MODEL_NAMES.includes(appName)) continue
    let mostRecent = 0
    let numRuns = 0
    for (let regionName of Object.keys(apps[appName])) {
      for (let uuid of Object.keys(apps[appName][regionName])) {
        numRuns++
        const run = apps[appName][regionName][uuid]
        if (run.timestamp > mostRecent) {
          mostRecent = run.timestamp
        }
      }
    }
    appData.push({
      appName,
      mostRecent,
      numRuns,
    })
  }
  return { props: { appData } }
}

const AppPage = ({ appData }) => {
  return (
    <Page title="Autumn Data">
      <h1>Autumn Results</h1>
      <List relaxed divided size="medium">
        <List.Item key="dhhs">
          <Header as="h3">
            <Link href="/dhhs" as={`/dhhs`}>
              <a>DHHS Reports</a>
            </Link>
          </Header>
        </List.Item>
        {appData
          .sort((a, b) => (a.appName > b.appName ? 1 : -1))
          .map(({ appName, mostRecent, numRuns }) => (
            <List.Item key={appName}>
              <Header as="h3">
                <Link href="/app/[appName]" as={`/app/${appName}`}>
                  <a>{appName.replace('_', ' ')}</a>
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
