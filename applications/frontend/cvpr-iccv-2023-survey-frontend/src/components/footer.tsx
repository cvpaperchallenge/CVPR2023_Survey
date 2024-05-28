'use-client'
import { MagnifyingGlassIcon, DotsHorizontalIcon, GitHubLogoIcon, TwitterLogoIcon, SunIcon, DesktopIcon, MoonIcon } from '@radix-ui/react-icons'
import {Link, Flex, IconButton, Text, SegmentedControl, Separator} from '@radix-ui/themes'

export default function Footer(){
  const year = new Date().getFullYear()

  return (
    <section style={{ flexShrink: 0, background: 'var(--accent-2)', minHeight: "30px" , padding: "30px 50px"}}>
      <Flex direction="row" justify="between">
        <Flex direction="column" gap="3">
          <Flex direction="row" width="100px" gap="3" style={{ alignItems: "center"}} >
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
          <Text color="gray" size="1">&copy; 2015-{year} cvpaper.challenge </Text>
        </Flex>
      </Flex>
    </section>
  )
}