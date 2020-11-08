import { useState } from 'react'
import { useRouter } from 'next/router'
import moment from 'moment'
import { Header } from 'semantic-ui-react'
import { Tab } from 'semantic-ui-react'
import { List } from 'semantic-ui-react'
import { Dropdown } from 'semantic-ui-react'
import styled from 'styled-components'

import { Page } from 'comps/page'
import { GitCommit } from 'comps/commit'

export async function getStaticPaths() {
  const data = await import('../../../../../../website.json')
  const { apps } = data.default
  const paths = []
  for (let appName of Object.keys(apps)) {
    for (let regionName of Object.keys(apps[appName])) {
      for (let uuid of Object.keys(apps[appName][regionName])) {
        paths.push({ params: { appName, regionName, uuid } })
      }
    }
  }
  return {
    paths,
    fallback: false,
  }
}

export async function getStaticProps({
  params: { appName, regionName, uuid },
}) {
  const data = await import('../../../../../../website.json')
  const { apps } = data.default

  const { id, timestamp, commit, files } = apps[appName][regionName][uuid]
  return { props: { id, timestamp, commit, files } }
}

const RunPage = ({ id, timestamp, commit, files }) => {
  const router = useRouter()
  const { appName, regionName, uuid } = router.query
  const dateStr = moment(timestamp, 'X').format('dddd, MMMM Do YYYY, h:mm a')
  const fs = files.map((f) => ({ ...f, filename: f.path.split('/').pop() }))
  const panes = [
    {
      menuItem: 'Data',
      render: () => (
        <Tab.Pane>
          <DataTab files={fs} />
        </Tab.Pane>
      ),
    },
    {
      menuItem: 'Plots',
      render: () => (
        <Tab.Pane>
          {' '}
          <PlotsTab files={fs} />
        </Tab.Pane>
      ),
    },
    {
      menuItem: 'Logs',
      render: () => (
        <Tab.Pane>
          {' '}
          <LogsTab files={fs} />
        </Tab.Pane>
      ),
    },
  ]
  return (
    <Page title="Autumn Data">
      <Header as="h1">
        {regionName.replace('-', ' ')} ({appName.replace('_', ' ')})
        <Header.Subheader>
          commit <GitCommit commit={commit} /> at {dateStr} <br />
          <a
            href={`https://s3.console.aws.amazon.com/s3/buckets/autumn-data/${id}/`}
          >
            {id}
          </a>
        </Header.Subheader>
      </Header>
      <Tab panes={panes} />
    </Page>
  )
}
export default RunPage

const DataTab = ({ files }) => {
  const folders = [
    {
      title: 'PowerBI Outputs',
      desc: 'Database files from PowerBI post processing',
      files: files.filter((f) => f.path.startsWith('data/powerbi')),
    },
    {
      title: 'Full Model Run Outputs',
      desc: 'Database files from full model runs',
      files: files.filter((f) => f.path.startsWith('data/full_model_runs')),
    },
    {
      title: 'Calibration Outputs',
      desc: 'Database files from calibration run',
      files: files.filter((f) => f.path.startsWith('data/calibration_outputs')),
    },
  ]

  return (
    <List>
      {folders.map(({ title, desc, files }) => (
        <List.Item key={title}>
          <List.Icon name="folder" />
          <List.Content>
            <List.Header>{title}</List.Header>
            <List.Description>{desc}</List.Description>
            <List.List>
              {files.map(({ filename, path, url }) => (
                <List.Item key={url}>
                  <List.Icon name="database" />
                  <List.Content>
                    <a href={url}>{getFileName(filename, path)}</a>
                  </List.Content>
                </List.Item>
              ))}
            </List.List>
          </List.Content>
        </List.Item>
      ))}
    </List>
  )
}

const getFileName = (filename, path) => {
  if (filename.endsWith('.feather')) {
    let [chain, fname] = path.split('/').slice(-2)
    chain = chain.replace('-', ' ')
    return (
      <span>
        <strong>{chain}:</strong> {fname}
      </span>
    )
  } else {
    return <span>{filename}</span>
  }
}

const PlotsTab = ({ files }) => {
  const [currentCat, setCat] = useState('loglikelihood')
  const plotFiles = files.filter((f) => f.path.startsWith('plots'))
  if (plotFiles.length < 1) {
    return <p>No plots present</p>
  }
  const categories = new Set()
  for (let file of plotFiles) {
    const parts = file.path.split('/')
    if (parts.length > 2) {
      file.category = parts[1]
    } else {
      file.category = 'loglikelihood'
    }
    categories.add(file.category)
  }
  const plotOptions = Array.from(categories).map((c) => ({
    key: c,
    text: c,
    value: c,
  }))
  const handleChange = (e, { value }) => setCat(value)
  return (
    <>
      <Dropdown
        fluid
        selection
        value={currentCat}
        placeholder="Select a plot type"
        onChange={handleChange}
        options={plotOptions}
      />
      {plotFiles
        .filter((p) => p.category === currentCat)
        .map(({ url }) => (
          <PlotImg key={url} src={url} />
        ))}
    </>
  )
}

const PlotImg = styled.img`
  width: 100%;
  padding: 1rem;
  max-width: 500px;
`

const LogsTab = ({ files }) => {
  const logFiles = files.filter((f) => f.path.startsWith('logs'))
  console.log(logFiles)
  return logFiles.map((f) => (
    <p key={f.url}>
      <a href={f.url}>{f.path}</a>
    </p>
  ))
}
