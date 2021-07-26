import { useState } from 'react'
import { useRouter } from 'next/router'
import moment from 'moment'
import { Header } from 'semantic-ui-react'
import { Tab } from 'semantic-ui-react'
import { List } from 'semantic-ui-react'
import { Image } from 'semantic-ui-react'
import { Dropdown } from 'semantic-ui-react'
import styled from 'styled-components'

import { Page } from 'comps/page'
import { GitCommit } from 'comps/commit'

export async function getStaticPaths() {
  const data = await import('../../../../../../../website.json')
  const { apps } = data.default
  const paths = []
  for (let appName of Object.keys(apps)) {
    for (let regionName of Object.keys(apps[appName])) {
      for (let uuid of Object.keys(apps[appName][regionName])) {
        // weird uuids for some reason
        // eg. 1622502753-003079g\\data\\calibration_outputs\\chain-0\\mcmc_run.parquet
        if (uuid.includes('\\')) continue
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
  const data = await import('../../../../../../../website.json')
  const { apps } = data.default

  const { id, timestamp, commit, files } = apps[appName][regionName][uuid]
  return { props: { id, timestamp, commit, files } }
}

const DashPage = ({ id, timestamp, commit, files }) => {
  const router = useRouter()
  const { appName, regionName, uuid } = router.query
  const dateStr = moment(timestamp, 'X').format('dddd, MMMM Do YYYY, h:mm a')
  const fs = files.map((f) => ({ ...f, filename: f.path.split('/').pop() }))
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
      <Tab.Pane>
          <PlotsTab files={fs} />
        </Tab.Pane>
    </Page>
  )
}
export default DashPage

const PlotsTab = ({ files }) => {
  const [currentCat, setCat] = useState('dash')
  const plotFilesPre = files.filter((f) => f.path.startsWith('plots'))
  const plotFiles = plotFilesPre.filter((f) => f.path.split('/')[1] != 'dashboard')
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
  return (
    <>

      {plotFiles
        .filter((p) => p.category === currentCat)
        .map(({ url }) => (
          <Image src={url}></Image>
        ))}
    </>
  )
}

