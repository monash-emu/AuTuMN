import Link from 'next/link'
import { Header } from 'semantic-ui-react'
import { List } from 'semantic-ui-react'
import styled from 'styled-components'
import moment from 'moment'

import { Page } from 'comps/page'

export async function getStaticProps() {
  const data = await import('../website.json')
  const { models, runs } = data.default
  const modelFacts = {}
  for (let model of models) {
    const modelRuns = Object.values(runs[model])
    const mostRecent = modelRuns.reduce(
      (a, v) => (a.timestamp > v.timestamp ? a : v),
      { timestamp: 0 }
    )
    modelFacts[model] = {
      numRuns: modelRuns.length,
      mostRecent,
    }
  }

  return { props: { models, modelFacts } }
}

const HomePage = ({ models, modelFacts }) => {
  return (
    <Page title="Autumn Data">
      <h1>COVID Models</h1>
      <List relaxed divided size="medium">
        {models.sort().map((m) => (
          <List.Item key={m}>
            <Header as="h3">
              <Link href="/model/[model]" as={`/model/${m}`}>
                <a>{m}</a>
              </Link>
            </Header>
            <List.Description>
              {modelFacts[m].numRuns} runs - last run was{' '}
              {moment(modelFacts[m].mostRecent.timestamp, 'X').fromNow()}
            </List.Description>
          </List.Item>
        ))}
      </List>
    </Page>
  )
}

export default HomePage
