import Link from 'next/link'
import { Header } from 'semantic-ui-react'
import { List } from 'semantic-ui-react'
import moment from 'moment'

import { Page } from 'comps/page'

export async function getStaticProps() {
  return {
    props: {
      files: [{ filename: 'foo.csv', url: 'https://google.com' }],
    },
  }
}

const DHHSPage = ({ files }) => {
  return (
    <Page title="DHHS Reporting">
      <h1>DHHS Reporting</h1>
      <List relaxed divided size="medium">
        {files.map(({ filename, url }) => (
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

export default DHHSPage
