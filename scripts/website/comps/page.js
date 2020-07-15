import Head from 'next/head'
import Link from 'next/link'
import { Menu, Container } from 'semantic-ui-react'

export const Page = ({ title, children }) => (
  <>
    <Head>
      <title>{title}</title>
      <link
        rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/semantic-ui@2.4.2/dist/semantic.min.css"
      />
    </Head>
    <Menu inverted>
      <Link href="/">
        <Menu.Item>Autumn Calibrations</Menu.Item>
      </Link>
    </Menu>
    <Container>{children}</Container>
  </>
)
