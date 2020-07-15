import Head from 'next/head'
import Link from 'next/link'
import { Menu, Container } from 'semantic-ui-react'
import styled from 'styled-components'

export const Page = ({ title, children }) => (
  <>
    <Head>
      <title>{title}</title>
      <link
        rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/semantic-ui@2.4.2/dist/semantic.min.css"
      />
      <link rel="shortcut icon" href="/favicon.ico" />
    </Head>
    <Menu inverted>
      <Link href="/">
        <Menu.Item>
          <TitleImage src="/maple-leaf.png" />
          Autumn Data
        </Menu.Item>
      </Link>
    </Menu>
    <Container>{children}</Container>
  </>
)

const TitleImage = styled.img`
  transform: translate(-3px, -3px);
`
