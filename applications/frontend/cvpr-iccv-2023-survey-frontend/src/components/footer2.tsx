'use client'
import { MagnifyingGlassIcon, DotsHorizontalIcon, GitHubLogoIcon, TwitterLogoIcon, SunIcon, DesktopIcon, MoonIcon } from '@radix-ui/react-icons'
import {Link, Flex, IconButton, Text, SegmentedControl, Separator} from '@radix-ui/themes'

export default function Footer(){
  const year = new Date().getFullYear()

  return (
    <section
      style={{
        background: 'var(--accent-2)',
    }}>
      <Flex
        direction="row"
        justify="between"
        align="start"
        style={{
          flexShrink: 0,
          padding: "30px 50px",
          margin: "0 auto",
          width: "1100px",
          maxWidth: "100%",
      }}>
        <Flex direction="row" align="center" gap="5" style={{paddingTop: "10px"}}>
          <Flex direction="column" align="start" gap="2" >
            <img
              src="/cc_logo_2white.png"
              alt="logo"
              style={{
                objectFit: 'cover',
                width: '240px',
                height: '23px',
              }}
            />
            <Text color="gray" size="2" style={{paddingLeft: "5px"}}>&copy; 2015-{year}</Text>
          </Flex>
          <Separator orientation="vertical" size="2"/>
          <Flex direction="row" align="center" gap="5">
            <Text color="gray" size="2">
              Supported by SSII 2024
            </Text>
          </Flex>
        </Flex>
        <Flex direction="row" align="center" gap="4" style={{ backgroundColor: 'var(--sage-a3)', borderRadius: 'var(--radius-3)', padding: '10px 30px'}}>
          <Flex direction="column" align="start" gap="2">
            <Text size="3">
              Developed by<br/>
            </Text>
            <Flex direction="column" align="start" gap="1">
              <Link href="https://github.com/YoshikiKubotani">
                <Flex direction="row" align="center" gap="1">
                  <GitHubLogoIcon />
                  <Text color="gray" size="2" weight="light">Yoshiki Kubotani</Text>
                </Flex>
              </Link>
              <Link href="https://github.com/gatheluck">
                <Flex direction="row" align="center" gap="1">
                  <GitHubLogoIcon />
                  <Text color="gray" size="2" weight="light">Yoshihiro Fukuhara</Text>
                </Flex>
              </Link>
              <Link href="https://github.com/Hina39">
                <Flex direction="row" align="center" gap="1">
                  <GitHubLogoIcon />
                  <Text color="gray" size="2" weight="light">Hina Otake</Text>
                </Flex>
              </Link>
            </Flex>
          </Flex>
          <img
            src="/forward-propergation-chan4.png"
            alt="jundenpa_chan"
            style={{
              objectFit: 'cover',
              width: '40px',
              height: '40px',
            }}
          />
        </Flex>
      </Flex>
    </section>
  )
}