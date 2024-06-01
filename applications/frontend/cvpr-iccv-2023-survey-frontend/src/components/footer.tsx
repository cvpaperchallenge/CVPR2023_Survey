'use client'
import Image from 'next/image'
import { GitHubLogoIcon } from '@radix-ui/react-icons'
import {Link, Flex, Text, Separator} from '@radix-ui/themes'

import "./footer-styles.css"

export default function Footer(){
  const year = new Date().getFullYear()

  return (
    <section
      style={{
        background: 'var(--accent-2)',
        minWidth: '340px',
        boxSizing: 'border-box',
    }}>
      <Flex
        direction="row"
        justify="between"
        align="start"
        minWidth="840px"
        maxWidth="100%"
        width="100vw"
        mx="auto"
        px="50px"
        py="30px"
        className='footer'
        style={{
          flexShrink: 0,
          boxSizing: 'border-box',
      }}>
        <Flex direction="row" align="center" gap="5" className="flex-structure" style={{paddingTop: "10px"}}>
          <Flex direction="column" align="start" gap="2" >
            <Image
              src="/cc_logo_2white.png"
              alt="logo"
              width={240}
              height={23}
              className='footer-logo'
              style={{
                objectFit: 'cover',
              }}
            />
            <Text color="gray" size="2" className='detailed-text' style={{paddingLeft: "5px"}}>&copy; 2015-{year}</Text>
          </Flex>
          <Separator orientation="horizontal" size="2" className='separator'/>
          <Flex direction="row" align="center" gap="5">
            <Text color="gray" size="2" className='detailed-text'>
              Supported by SSII 2024
            </Text>
          </Flex>
        </Flex>
        <Flex direction="row" align="center" gap="4" className="developer-box" style={{ backgroundColor: 'var(--sage-a3)', borderRadius: 'var(--radius-3)'}}>
          <Flex direction="column" align="start" gap="2" width="140px">
            <Text size="3" className='developer-heading'>
              Developed by<br/>
            </Text>
            <Flex direction="column" align="start" gap="1">
              <Link href="https://github.com/YoshikiKubotani">
                <Flex direction="row" align="center" gap="1">
                  <GitHubLogoIcon />
                  <Text color="gray" size="2" weight="light" className='developer-text'>Yoshiki Kubotani</Text>
                </Flex>
              </Link>
              <Link href="https://github.com/gatheluck">
                <Flex direction="row" align="center" gap="1">
                  <GitHubLogoIcon />
                  <Text color="gray" size="2" weight="light" className='developer-text'>Yoshihiro Fukuhara</Text>
                </Flex>
              </Link>
              <Link href="https://github.com/Hina39">
                <Flex direction="row" align="center" gap="1">
                  <GitHubLogoIcon />
                  <Text color="gray" size="2" weight="light" className='developer-text'>Hina Otake</Text>
                </Flex>
              </Link>
            </Flex>
          </Flex>
          <Image
            src="/forward-propergation-chan4.png"
            alt="jundenpa_chan"
            width={40}
            height={40}
            className='forward-propagation-chan-image'
            style={{
              objectFit: 'cover',
            }}
          />
        </Flex>
      </Flex>
    </section>
  )
}