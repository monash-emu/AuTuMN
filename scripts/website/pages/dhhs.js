import Link from 'next/link'
import { Header } from 'semantic-ui-react'
import { List } from 'semantic-ui-react'
import moment from 'moment'

import { Page } from 'comps/page'

export async function getStaticProps() {
  const data = await import('../website.json')
  const { dhhs } = data.default
  return { props: { files: dhhs } }
}

const DHHSPage = ({ files }) => {
  return (
    <Page title="DHHS Reporting">
      <h1>DHHS Reporting</h1>
      <List relaxed divided size="medium">
        {files
          .sort((a, b) => (fileToDate(a) > fileToDate(b) ? 1 : -1))
          .map(({ filename, url }) => (
            <List.Item key={url}>
              <List.Icon name="database" />
              <List.Content>
                <a href={url}>{filename}</a>
              </List.Content>
            </List.Item>
          ))}
      </List>
    </Page>
  )
}

const fileToDate = ({ filename }) => {
  let dateStr = filename.split('.')[0].split('-').slice(3, 8).join('-')
  dateStr =
    dateStr.split('T')[0] + 'T' + dateStr.split('T')[1].split('-').join(':')
  return Date.parse(dateStr)
}

export default DHHSPage
