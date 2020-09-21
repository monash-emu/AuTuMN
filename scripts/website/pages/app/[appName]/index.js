import Link from 'next/link'
import { Header } from 'semantic-ui-react'
import { List } from 'semantic-ui-react'
import styled from 'styled-components'
import moment from 'moment'
import { useRouter } from 'next/router'

import { Page } from 'comps/page'

export async function getStaticPaths() {
  const data = await import('../../../website.json')
  const { apps } = data.default
  return {
    paths: Object.keys(apps).map((appName) => ({ params: { appName } })),
    fallback: false,
  }
}
export async function getStaticProps({ params: { appName } }) {
  const data = await import('../../../website.json')
  const { apps } = data.default
  const regionData = []
  for (let regionName of Object.keys(apps[appName])) {
    if (regionName == 'test') {
      continue
    }
    let mostRecent = 0
    let numRuns = 0
    for (let uuid of Object.keys(apps[appName][regionName])) {
      numRuns++
      const run = apps[appName][regionName][uuid]
      if (run.timestamp > mostRecent) {
        mostRecent = run.timestamp
      }
    }
    regionData.push({
      regionName,
      mostRecent,
      numRuns,
    })
  }
  return { props: { regionData } }
}

const RegionPage = ({ regionData }) => {
  const router = useRouter()
  const { appName } = router.query
  return (
    <Page title="Autumn Data">
      <h1 className="ui header">{appName.replace('_', ' ')} Results</h1>
      <List relaxed divided size="medium">
        {regionData
          .sort((a, b) => (a.regionName > b.regionName ? 1 : -1))
          .map(({ regionName, mostRecent, numRuns }) => (
            <List.Item key={regionName}>
              <Header as="h3">
                <Link
                  href="/app/[appName]/region/[regionName]"
                  as={`/app/${appName}/region/${regionName}`}
                >
                  <a>{regionName.replace('-', ' ')}</a>
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

export default RegionPage
