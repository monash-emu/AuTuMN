import Link from 'next/link'
import { Header } from 'semantic-ui-react'
import { List } from 'semantic-ui-react'
import moment from 'moment'

import { fileToDate } from 'utils'
import { Page } from 'comps/page'
import { useRouter } from 'next/router'
import styled from 'styled-components'

export async function getStaticPaths() {
  const data = await import('../../website.json')
  const { reports } = data.default
  return {
    paths: Object.keys(reports).map((reportName) => ({
      params: { reportName },
    })),
    fallback: false,
  }
}

export async function getStaticProps({ params: { reportName } }) {
  const data = await import('../../website.json')
  const { reports } = data.default
  return { props: reports[reportName] }
}

const ReportPage = ({ title, description, files }) => {
  const router = useRouter()
  const { reportName } = router.query

  return (
    <Page title={`${title} Report`}>
      <Header as="h1">
        {title} Report{' '}
        <Header.Subheader>
          {description} (
          <a
            href={`https://s3.console.aws.amazon.com/s3/buckets/autumn-data/${reportName}/`}
          >
            report bucket
          </a>
          )
        </Header.Subheader>
      </Header>

      <List relaxed divided size="large">
        {files
          .map((f) => ({ ...f, timestamp: fileToDate(f) }))
          .sort((a, b) => (a.timestamp > b.timestamp ? 1 : -1))
          .map(({ filename, url, timestamp }) => {
            return (
              <List.Item key={url}>
                <List.Icon name="file alternate outline" />
                <List.Content>
                  <a href={url}>{filename}</a>
                </List.Content>
                <Description>
                  Created {moment(timestamp, 'X').fromNow()}
                </Description>
              </List.Item>
            )
          })}
      </List>
    </Page>
  )
}

const Description = styled(List.Description)`
  padding: 2px 0 2px 25px !important;
  font-size: 14px !important;
`

export default ReportPage
