import Link from 'next/link'
import { useRouter } from 'next/router'
import { Page } from 'comps/page'
import { Header } from 'semantic-ui-react'
import { List } from 'semantic-ui-react'
import { Icon, Label } from 'semantic-ui-react'
import moment from 'moment'

import { GitCommit } from 'comps/commit'

export async function getStaticPaths() {
  const data = await import('../../../website.json')
  return {
    paths: data.default.models.map((model) => ({ params: { model } })),
    fallback: false,
  }
}
export async function getStaticProps({ params: { model } }) {
  const data = await import('../../../website.json')
  const { runs, models } = data.default
  const modelRuns = Object.values(runs[model])
    .sort((a, b) => b.timestamp - a.timestamp)
    .map(({ id, model, timestamp, commit, files }) => ({
      id,
      model,
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

  return { props: { modelRuns } }
}

const ModelPage = ({ modelRuns }) => {
  const router = useRouter()
  const { model } = router.query
  return (
    <Page title={`Autumn Data - ${model}`}>
      <Header as="h1">
        {model}
        <Header.Subheader>{modelRuns.length} runs total</Header.Subheader>
      </Header>
      <List relaxed divided size="medium">
        {modelRuns.map((mr) => (
          <ModelRunListItem key={mr.id} {...mr} />
        ))}
      </List>
    </Page>
  )
}

const ModelRunListItem = ({
  id,
  model,
  timestamp,
  commit,
  isPowerBI,
  isFull,
  isCalib,
}) => {
  return (
    <List.Item key={id}>
      <Link href="/model/[model]/run/[id]" as={`/model/${model}/run/${id}`}>
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

export default ModelPage
