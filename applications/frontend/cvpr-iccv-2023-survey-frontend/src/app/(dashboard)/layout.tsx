import { MagnifyingGlassIcon, DotsHorizontalIcon } from '@radix-ui/react-icons'
import {Box, Flex, Text, TextField, Tabs} from '@radix-ui/themes'

export default function PaperListLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <Flex direction="column" align="stretch" gap="9">
      <section>Header</section>
      <Flex direction="column" align="center" gap="0">
        {children}
      </Flex>
      <section>Footer</section>
    </Flex>
  )
}