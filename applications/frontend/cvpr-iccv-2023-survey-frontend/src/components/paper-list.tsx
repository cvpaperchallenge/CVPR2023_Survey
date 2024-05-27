import {Text, Flex, Card, Box, Avatar, Skeleton} from '@radix-ui/themes'
import Link from 'next/link'

import { Paper } from '../libs/types';

interface PaperListProps {
  papers: Paper[];
  conferenceName: string;
}

export default function PaperList({
  papers,
  conferenceName,
}: PaperListProps) {
  const numSkeletonCard = 3

  if (papers.length === 0) return (
    <Flex direction="column" align="center" gap="3" width="100%">
      {[...Array(numSkeletonCard)].map((_, index) => (
        <Skeleton key={index}>
          <Box width="90%">
            <Card>
              <Text as="div" size="2" weight="bold">
                Dummy Title
              </Text>
              <Text as="div" color="gray" size="2">
                Dummy Contents
              </Text>
            </Card>
          </Box>
        </Skeleton>
      ))}
    </Flex>
  )

  return (
    <Flex direction="column" align="center" gap="3" width="100%">
      {papers.map((paper, index) => (
        <Box width="90%" key={index}>
          <Card asChild>
            <Link href={`/details?conference=${conferenceName}&id=${index}`}>
              <Flex gap="3" direction="row" align="center">
                <Avatar fallback={index} />
                <Box>
                  <Text as="div" size="2" weight="bold">
                    {paper.title}
                  </Text>
                  <Text as="div" color="gray" size="2">
                    {paper.author}
                  </Text>
                </Box>
              </Flex>
            </Link>
          </Card>
        </Box>
      ))}
    </Flex>
  )
}
