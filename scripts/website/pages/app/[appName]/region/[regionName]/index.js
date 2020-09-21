import Link from 'next/link'
import { useRouter } from 'next/router'
import { Page } from 'comps/page'
import { Header } from 'semantic-ui-react'
import { List } from 'semantic-ui-react'
import { Icon, Label } from 'semantic-ui-react'
import moment from 'moment'

import { GitCommit } from 'comps/commit'

export async function getStaticPaths() {
  const data = await import('../../../../../website.json')
  const { apps } = data.default
  const paths = []
  for (let appName of Object.keys(apps)) {
    for (let regionName of Object.keys(apps[appName])) {
      paths.push({ params: { appName, regionName } })
    }
  }
  return {
    paths,
    fallback: false,
  }
}
export async function getStaticProps({ params: { appName, regionName } }) {
  const data = await import('../../../../../website.json')
  const { apps } = data.default
  const regionRuns = []
  for (let uuid of Object.keys(apps[appName][regionName])) {
    const run = apps[appName][regionName][uuid]
    regionRuns.push(run)
  }
  const runs = regionRuns
    .sort((a, b) => b.timestamp - a.timestamp)
    .map(({ id, app, region, timestamp, commit, files }) => ({
      id,
      app,
      region,
      timestamp,
      commit,
      isPowerBI:
        files.filter((f) => f.path.startsWith('data/powerbi')).length > 0,
      isFull:
        files.filter((f) => f.path.startsWith('data/full_model_runs')).length >
        0,
      isCalib:
        files.filter((f) => f.path.startsWith('data/calibration_outputs'))
          .length > 0,
    }))

  return { props: { runs } }
}

const RegionPage = ({ runs }) => {
  const router = useRouter()
  const { appName, regionName } = router.query
  return (
    <Page title="Autumn Data">
      <Header as="h1">
        {regionName.replace('-', ' ')} ({appName.replace('_', ' ')})
        <Header.Subheader>{runs.length} runs total</Header.Subheader>
      </Header>
      <List relaxed divided size="medium">
        {runs.map((mr) => (
          <RunListItem key={mr.id} {...mr} />
        ))}
      </List>
    </Page>
  )
}

const RunListItem = ({
  id,
  app,
  region,
  timestamp,
  commit,
  isPowerBI,
  isFull,
  isCalib,
}) => {
  let uuid // timestamp-commit
  if (id.includes('/')) {
    // New run id format
    // app/region/timestamp/commit
    uuid = id.split('/').slice(-2).join('-')
  } else {
    // Old run id format
    // modelname-timestamp-commit
    uuid = id.split('-').slice(-2).join('-')
  }
  return (
    <List.Item key={id}>
      <Link
        href="/app/[appName]/region/[regionName]/run/[uuid]"
        as={`/app/${app}/region/${region}/run/${uuid}`}
      >
        <List.Header as="a">{id}</List.Header>
      </Link>
      <List.Description>
        Git commit <GitCommit commit={commit} />, run{' '}
        {moment(timestamp, 'X').fromNow()}
        <List.Content floated="right">
          {isPowerBI && (
            <Label color="green">
              <Icon name="database" /> PowerBI
            </Label>
          )}
          {isFull && (
            <Label color="teal">
              <Icon name="database" /> Full Runs
            </Label>
          )}
          {isCalib && (
            <Label color="blue">
              <Icon name="database" /> Calibration
            </Label>
          )}
        </List.Content>
      </List.Description>
    </List.Item>
  )
}

export default RegionPage
