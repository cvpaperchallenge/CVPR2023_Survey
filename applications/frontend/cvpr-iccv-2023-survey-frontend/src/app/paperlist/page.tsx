'use client'
import { cache, useRef, useState, type RefObject, useEffect } from 'react'
import { toast } from 'sonner'
import { MagnifyingGlassIcon, DotsHorizontalIcon } from '@radix-ui/react-icons'
import {Heading, Text, Flex, Tabs, TextField, Box, Separator} from '@radix-ui/themes'

import { Paper, FetchResult } from '../../libs/types'
import { getPaperLists, searchPapers } from '../../libs/fetch'
import PaperListBoard from '../../components/paper-list-board'

const handleFetchResult = <T,>(result: FetchResult<T>, errorMessage: string): T | [] => {
  if (result.error) {
    toast.error(result.error || errorMessage);
    return [];
  }
  return result.data || [];
};

export const loadPaperLists = cache(async (conference: string) => {
  const result = await getPaperLists(conference);
  return handleFetchResult<Paper[]>(result, 'Failed to fetch papers');
});

export const loadSearchResults = async (query: string, conference: string) => {
  const result = await searchPapers(query, conference);
  return handleFetchResult<Paper[]>(result, 'Failed to fetch search results');
};


export default function PaperList() {
  const inputRef: RefObject<HTMLInputElement> = useRef(null)

  const [papers, setPapers] = useState<Paper[]>([])
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

  const handleInputChange = (event: any) => {
    setSearchText(event.target.value);
  };

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
    <Flex direction="column" align="center" gap="3" width="100%">
      <Flex direction="column" align="center" gap="1" width="100%">
        <Heading size="7">Papers List</Heading>
        <Separator my="2" size="3"/>
      </Flex>
      <Box width="100%">
        <TextField.Root
          ref={inputRef}
          placeholder="Search papers…"
          size="3"
          value={searchText}
          onKeyDown={handleKeyDown}
          onChange={handleInputChange}
        >
          <TextField.Slot>
            <MagnifyingGlassIcon height="16" width="16" />
          </TextField.Slot>
        </TextField.Root>
        <PaperListBoard
          papers={papers}
          conferenceName={conferenceName}
          setConferenceName={setConferenceName}
        />
      </Box>
    </Flex>
  )
}