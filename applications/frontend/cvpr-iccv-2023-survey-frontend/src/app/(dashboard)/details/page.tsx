'use client'
import { cache, useState, useEffect } from 'react'
import { toast } from 'sonner'
import { FileIcon, GlobeIcon } from '@radix-ui/react-icons'
import {Heading, Text, Flex, Link, DataList, Separator, Button, Box} from '@radix-ui/themes'
import { useSearchParams, useRouter } from 'next/navigation'

import { PaperDetails } from '@/libs/types'
import { getPaperDetails, handleFetchResult } from '@/libs/fetch'
import "./styles.css"

const loadPaperDetails = cache(async (conference: string, id: string) => {
  const result = await getPaperDetails(conference, id);
  return handleFetchResult<PaperDetails>(result, 'Failed to fetch paper details');
})

export default function PaperInfo() {
  const router = useRouter()
  const searchParams = useSearchParams()

  const [paperDetails, setPaperDetails] = useState<PaperDetails | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const conference= searchParams.get('conference')
  const paperId= searchParams.get('id')

  useEffect(() => {
    if (!conference || !paperId) {
      toast.error('Invalid URL');
      router.push('/list')
    }
    else {
      const fetchPaperDetails = async () => {
        const paperDetails = await loadPaperDetails(conference, paperId)
        if (!paperDetails) {
          router.push('/list')
        }
        else {
          setPaperDetails(paperDetails)
          setIsLoading(false)
        }
      }
      fetchPaperDetails()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (isLoading || !paperDetails) {
    return (
      <Flex justify="center" align="center" gap="3" width="100%">
        <Text>Loading...</Text>
      </Flex>
    )
  }

  return (
    <Flex direction="column" align="center" gap="7" width="100%" maxWidth="1000px" minWidth="350px">
      <Box px="3">
        <Heading size="6">{paperDetails.paperInfo.title}</Heading>
      </Box>
      <Flex direction="column" gap="9" align="stretch" width="80%">
        <Flex direction="column" align="start" gap="3">
          <Flex direction="column" align="stretch" gap="0">
            <Flex direction="row" align="end" gap="2">
              <Heading size="5" className="heading">基本情報</Heading>
              <Text color="gray" size="3" weight="light" trim="end" className="heading-sub" style={{paddingBottom: "4px"}}>
                / Basic Information
              </Text>
            </Flex>
            <Separator my="2" size="4"/>
          </Flex>
          <Box p="4" style={{ backgroundColor: 'var(--gray-a2)', borderRadius: 'var(--radius-3)' }}>
            <DataList.Root size="2" orientation={{ initial: 'vertical', sm: 'horizontal' }}>
              <DataList.Item>
                <DataList.Label minWidth="88px">ID</DataList.Label>
                <DataList.Value>{paperId}</DataList.Value>
              </DataList.Item>
              <DataList.Item>
                <DataList.Label minWidth="88px">Authors</DataList.Label>
                <DataList.Value>{paperDetails.paperInfo.authors.join(", ")}</DataList.Value>
              </DataList.Item>
              <DataList.Item>
                <DataList.Label minWidth="88px">Abstract</DataList.Label>
                <DataList.Value>
                  {paperDetails.paperInfo.abstract}
                </DataList.Value>
              </DataList.Item>
              <DataList.Item>
                <DataList.Label minWidth="88px">Link</DataList.Label>
                <DataList.Value>
                  <Flex direction="row" gap="4">
                    <Link target="_blank" href={paperDetails.paperInfo.cvfLink}>
                      <Button variant="soft" size="1">
                        <GlobeIcon /> CVF
                      </Button>
                    </Link>
                    <Link target="_blank" href={paperDetails.paperInfo.pdfLink}>
                      <Button variant="soft" size="1" color="tomato">
                        <FileIcon /> PDF
                      </Button>
                    </Link>
                  </Flex>
                </DataList.Value>
              </DataList.Item>
            </DataList.Root>
          </Box>
        </Flex>
        <Flex direction="column" align="start" gap="3">
          <Flex direction="column" align="stretch" gap="0">
            <Flex direction="row" align="end" gap="2">
              <Heading size="5" className="heading">どんなもの？</Heading>
              <Text color="gray" size="3" weight="light" trim="end" className="heading-sub" style={{paddingBottom: "4px"}}>
                / Outline
              </Text>
            </Flex>
            <Separator my="2" size="4"/>
          </Flex>
          <Box px="5" py="4" style={{ backgroundColor: 'var(--gray-a2)', borderRadius: 'var(--radius-3)' }}>
            <Text size="3" wrap="pretty" className="contents">{paperDetails.summary.outline}</Text>
          </Box>
        </Flex>
        <Flex direction="column" align="start" gap="3">
          <Flex direction="column" align="stretch" gap="0">
            <Flex direction="row" align="end" gap="2">
              <Heading size="5" className="heading">先行研究と比べてどこがすごい？</Heading>
              <Text color="gray" size="3" weight="light" trim="end" className="heading-sub" style={{paddingBottom: "4px"}}>
                / Contribution
              </Text>
            </Flex>
            <Separator my="2" size="4"/>
          </Flex>
          <Box px="5" py="4" style={{ backgroundColor: 'var(--gray-a2)', borderRadius: 'var(--radius-3)' }}>
            <Text size="3" wrap="pretty" className="contents">{paperDetails.summary.contribution}</Text>
          </Box>
        </Flex>
        <Flex direction="column" align="start" gap="3">
          <Flex direction="column" align="stretch" gap="0">
            <Flex direction="row" align="end" gap="2">
              <Heading size="5" className="heading">技術や手法のキモはどこ？</Heading>
              <Text color="gray" size="3" weight="light" trim="end" className="heading-sub" style={{paddingBottom: "4px"}}>
                / Methods
              </Text>
            </Flex>
            <Separator my="2" size="4"/>
          </Flex>
          <Box px="5" py="4" style={{ backgroundColor: 'var(--gray-a2)', borderRadius: 'var(--radius-3)' }}>
            <Text size="3" wrap="pretty" className="contents">{paperDetails.summary.method}</Text>
          </Box>
        </Flex>
        <Flex direction="column" align="start" gap="3">
          <Flex direction="column" align="stretch" gap="0">
            <Flex direction="row" align="end" gap="2">
              <Heading size="5" className="heading">どうやって有効だと検証した？</Heading>
              <Text color="gray" size="3" weight="light" trim="end" className="heading-sub" style={{paddingBottom: "4px"}}>
                / Evaluation
              </Text>
            </Flex>
            <Separator my="2" size="4"/>
          </Flex>
          <Box px="5" py="4" style={{ backgroundColor: 'var(--gray-a2)', borderRadius: 'var(--radius-3)' }}>
            <Text size="3" wrap="pretty" className="contents">{paperDetails.summary.evaluation}</Text>
          </Box>
        </Flex>
        <Flex direction="column" align="start" gap="3">
          <Flex direction="column" align="stretch" gap="0">
            <Flex direction="row" align="end" gap="2">
              <Heading size="5" className="heading">議論はある？</Heading>
              <Text color="gray" size="3" weight="light" trim="end" className="heading-sub" style={{paddingBottom: "4px"}}>
                / Discussion
              </Text>
            </Flex>
            <Separator my="2" size="4"/>
          </Flex>
          <Box px="5" py="4" style={{ backgroundColor: 'var(--gray-a2)', borderRadius: 'var(--radius-3)' }}>
            <Text size="3" wrap="pretty" className="contents">{paperDetails.summary.discussion}</Text>
          </Box>
        </Flex>
      </Flex>
    </Flex>
  )
}