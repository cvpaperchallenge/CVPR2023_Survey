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
        align="center"
        style={{
          flexShrink: 0,
          padding: "30px 50px",
          margin: "0 auto",
          width: "1100px",
          maxWidth: "100%",
      }}>
        <Flex direction="column" gap="3" >
          <Flex direction="row" align="center" gap="3">
            <IconButton variant="ghost" asChild>
              <Link href="https://github.com/cvpaperchallenge">
                <GitHubLogoIcon style={{ width: '24px', height: '24px' }} />
              </Link>
            </IconButton>
            <Separator orientation="vertical" size="1"/>
            <IconButton variant="ghost" asChild>
              <Link href="https://twitter.com/CVpaperChalleng">
                <TwitterLogoIcon style={{ width: '24px', height: '24px' }} />
              </Link>
            </IconButton>
          </Flex>
          <Text color="gray" size="1">&copy; 2015-{year} cvpaper.challenge</Text>
        </Flex>
        <Flex direction="row" align="center" gap="3">
          <Flex direction="row" align="center" gap="5">
            <Flex direction="column" align="start" gap="2" minHeight="84px">
              <Text color="gray" size="2">
                Supported by <br/> SSII 2024
              </Text>
            </Flex>
            <Separator orientation="vertical" size="3"/>
            <Flex direction="column" align="start" gap="2">
              <Flex direction="column" align="start" gap="2">
                <Text color="gray" size="2">
                  Developed by<br/>
                </Text>
                <Flex direction="column" align="start" gap="1">
                  <Link href="https://github.com/YoshikiKubotani">
                    <Flex direction="row" justify="center" gap="1">
                      <GitHubLogoIcon />
                      <Text color="gray" size="1" weight="light">Yoshiki Kubotani</Text>
                    </Flex>
                  </Link>
                  <Link href="https://github.com/gatheluck">
                    <Flex direction="row" justify="center" gap="1">
                      <GitHubLogoIcon />
                      <Text color="gray" size="1" weight="light">Yoshihiro Fukuhara</Text>
                    </Flex>
                  </Link>
                  <Link href="https://github.com/Hina39">
                    <Flex direction="row" justify="center" gap="1">
                      <GitHubLogoIcon />
                      <Text color="gray" size="1" weight="light">Hina Otake</Text>
                    </Flex>
                  </Link>
                </Flex>
              </Flex>
            </Flex>
          </Flex>
          <img
            src="/forward-propergation-chan4.png"
            alt="jundenpa_chan"
            style={{
              objectFit: 'cover',
              width: '40px',
              height: '40px',
              borderRadius: 'var(--radius-2)',
            }}
          />
        </Flex>
      </Flex>
    </section>
  )
}