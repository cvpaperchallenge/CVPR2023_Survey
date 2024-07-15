'use client'
import { cache, useRef, useState, type RefObject, useEffect } from 'react'
import { toast } from 'sonner'
import { MagnifyingGlassIcon } from '@radix-ui/react-icons'
import {Heading, Flex, TextField, Box, Separator} from '@radix-ui/themes'

import { PaperInfo } from '@/libs/types'
import { getPaperList, searchPapers, handleFetchArrayResult } from '@/libs/fetch'
import PaperListBoard from '@/components/paper-list-board'

const loadPaperLists = cache(async (conference: string) => {
  const result = await getPaperList(conference);
  return handleFetchArrayResult<PaperInfo[]>(result, 'Failed to fetch papers');
});

const loadSearchResults = async (query: string, conference: string) => {
  const result = await searchPapers(query, conference);
  return handleFetchArrayResult<PaperInfo[]>(result, 'Failed to fetch search results');
};


export default function PaperList() {
  const inputRef: RefObject<HTMLInputElement> = useRef(null)

  const [papers, setPapers] = useState<PaperInfo[]>([])
  const [searchText, setSearchText] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [conferenceName, setConferenceName] = useState('cvpr-2023')

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus()
    }
  }
  , [])

  useEffect(() => {
    const fetchPapers = async () => {
      setPapers(await loadPaperLists(conferenceName))
    }
    fetchPapers()
  }, [conferenceName])

  const handleSubmit = async (event: any) => {
    event.preventDefault();
    setIsSubmitting(true);

    const query = searchText.trim()
    setSearchText('')
    if (!query) {
      setIsSubmitting(false)
      return
    }

    setPapers(await loadSearchResults(query, conferenceName))
    setIsSubmitting(false)
  };

  const handleKeyDown = (
    event: React.KeyboardEvent<HTMLInputElement>
  ): void => {
    if (
      event.key === 'Enter' &&
      !isSubmitting
    ) {
      handleSubmit(event)
    }
  }

  return (
    <Flex direction="column" align="stretch" gap="3" width="80%" maxWidth="1000px" minWidth="350px">
      <Flex direction="column" align="center" gap="1" width="100%">
        <Heading size="7">Paper List</Heading>
        <Separator my="2" size="3"/>
      </Flex>
      <Box width="100%">
        {/* <TextField.Root
          ref={inputRef}
          placeholder="Search papers…"
          size="3"
          value={searchText}
          onKeyDown={handleKeyDown}
          onChange={(e) => setSearchText(e.target.value)}
        >
          <TextField.Slot>
            <MagnifyingGlassIcon height="16" width="16" />
          </TextField.Slot>
        </TextField.Root> */}
        <PaperListBoard
          papers={papers}
          conferenceName={conferenceName}
          setConferenceName={setConferenceName}
        />
      </Box>
    </Flex>
  )
}