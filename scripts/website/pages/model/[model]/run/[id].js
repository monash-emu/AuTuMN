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
  const data = await import('../../../../website.json')
  const { runs, models } = data.default
  const runIds = models
    .map((m) => Object.values(runs[m]))
    .reduce((a, b) => [...a, ...b], [])
  return {
    paths: runIds.map(({ id, model }) => ({ params: { id, model } })),
    fallback: false,
  }
}

export async function getStaticProps({ params: { id, model } }) {
  const data = await import('../../../../website.json')
  const { runs } = data.default
  const { timestamp, commit, files } = runs[model][id]
  return { props: { timestamp, commit, files } }
}

const RunPage = ({ timestamp, commit, files }) => {
  const router = useRouter()
  const { id, model } = router.query
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
        {model}
        <Header.Subheader>
          commit <GitCommit commit={commit} /> at {dateStr} ({id})
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
            <List.Description>
              {desc} -{' '}
              <a href="#" onClick={downloadAll(files)}>
                download all
              </a>
            </List.Description>
            <List.List>
              {files.map(({ filename, url }) => (
                <List.Item key={url}>
                  <List.Icon name="database" />
                  <List.Content>
                    <a href={url}>{filename}</a>
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

const downloadAll = (files) => (e) => {
  // FIXME: Broken
  alert('This is broken, will fix later')
  return
  e.preventDefault()
  const link = document.createElement('a')
  link.style.display = 'none'
  document.body.appendChild(link)
  for (let { filename, url } of files) {
    console.log('DOWNLOAD', filename, url)
    link.setAttribute('download', filename)
    link.setAttribute('href', url)
    link.click()
  }
  document.body.removeChild(link)
}
