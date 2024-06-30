'use client'
import { Table } from '@tanstack/react-table'
import { useState, useMemo } from 'react'
import {
  RxCheck,
  RxCaretLeft,
  RxCaretRight,
  RxMagnifyingGlass,
  RxDoubleArrowLeft,
  RxDoubleArrowRight,
} from 'react-icons/rx'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandList,
  CommandSeparator,
} from '@/components/ui/command'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { Separator } from '@/components/ui/separator'

import { cn } from '@/lib/utils'

interface DataTableFacetedFilterProps<TData> {
  table: Table<TData>
  columnName: string
}

export function DataTableFacetedFilter<TData>({
  table,
  columnName,
}: DataTableFacetedFilterProps<TData>) {
  const column = table.getColumn(columnName)
  // if (!column) return null

  // Create a map object to store the frecency of each option
  const optionFrequency = useMemo(() => {
    const frequencyMap: Map<string, number> = new Map()
    if (columnName === 'authors') {
      // Iterate over the original map object having the authors array as keys
      column?.getFacetedUniqueValues().forEach((frequency, authors) => {
        // Iterate over the authors array
        (authors as string[]).forEach((author: string) => {
          // If the author is already in the map object, increment its frecency
          if (frequencyMap.has(author)) {
            frequencyMap.set(author, frequencyMap.get(author)! + frequency)
          } else {
            // Otherwise, set the frecency to the frequency
            frequencyMap.set(author, frequency)
          }
        })
      })
      return frequencyMap
    } else {
      return column?.getFacetedUniqueValues()
    }
  }, [column, columnName])

  const options: string[] = useMemo(
    () => Array.from(optionFrequency?.keys() ?? []) as string[],
    [optionFrequency]
  )
  const selectedValues = new Set<string>(column?.getFilterValue() as string[])

  const [searchTerm, setSearchTerm] = useState('')
  const [currentPage, setCurrentPage] = useState(1)
  const itemsPerPage = 20

  const filteredOptions = useMemo(() => {
    return options
      .sort()
      .filter((option: string) =>
        option.toLowerCase().includes(searchTerm.toLowerCase())
      )
  }, [options, searchTerm])

  const paginatedOptions = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage
    return filteredOptions.slice(startIndex, startIndex + itemsPerPage)
  }, [filteredOptions, currentPage, itemsPerPage])

  const totalPages = Math.ceil(filteredOptions.length / itemsPerPage)

  const handleSearchTermChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setSearchTerm(event.target.value)
    setCurrentPage(1)
  }

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          className="h-8 w-fit rounded-sm border-dashed bg-[var(--teal-4)]"
          size="sm"
          variant="outline"
        >
          {String(column?.columnDef.header)}
          {selectedValues?.size > 0 && (
            <>
              <Separator className="mx-2 h-4" orientation="vertical" />
              <Badge
                className="rounded-sm bg-[var(--teal-9)] px-1 font-normal sm:hidden"
                variant="default"
              >
                {selectedValues.size}
              </Badge>
              <div className="hidden space-x-1 sm:flex">
                {selectedValues.size > 2 ? (
                  <Badge
                    className="rounded-sm bg-[var(--teal-9)] px-1 font-normal"
                    variant="default"
                  >
                    {selectedValues.size} selected
                  </Badge>
                ) : (
                  Array.from(options)
                    .sort()
                    .filter((option: string) => selectedValues.has(option))
                    .map((option: string) => (
                      <Badge
                        className="rounded-sm bg-[var(--teal-9)] px-1 font-normal"
                        key={option}
                        variant="default"
                      >
                        {option}
                      </Badge>
                    ))
                )}
              </div>
            </>
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-[250px] p-0">
        <Command>
          <div className="flex items-center border-b px-3">
            <RxMagnifyingGlass className="mr-2 size-4 shrink-0 opacity-50" />
            <input
              className="flex h-11 w-full rounded-md bg-transparent py-3 text-sm font-medium outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-50"
              onChange={handleSearchTermChange}
              placeholder="Search..."
              value={searchTerm}
            />
          </div>
          <CommandList>
            <CommandEmpty>No results found.</CommandEmpty>
            <CommandGroup>
              {paginatedOptions.map((option) => {
                const isSelected = selectedValues.has(option)
                return (
                  <CommandItem
                    key={option}
                    onSelect={() => {
                      if (isSelected) {
                        selectedValues.delete(option)
                      } else {
                        selectedValues.add(option)
                      }
                      const filterValues = Array.from(selectedValues)
                      column?.setFilterValue(
                        filterValues.length ? filterValues : undefined
                      )
                    }}
                  >
                    <div
                      className={cn(
                        'mr-2 flex h-4 w-4 items-center justify-center rounded-sm border border-primary',
                        isSelected
                          ? 'bg-primary text-primary-foreground'
                          : 'opacity-50 [&_svg]:invisible'
                      )}
                    >
                      <RxCheck className={cn('h-4 w-4')} />
                    </div>
                    <span>{option}</span>
                    <span className="ml-auto flex size-4 items-center justify-center font-mono text-xs">
                      {optionFrequency?.get(option)}
                    </span>
                  </CommandItem>
                )
              })}
            </CommandGroup>
          </CommandList>
          {selectedValues.size > 0 && (
            <>
              <CommandSeparator />
              <CommandGroup>
                <CommandItem
                  className="justify-center text-center"
                  onSelect={() => column?.setFilterValue(undefined)}
                >
                  Clear filters
                </CommandItem>
              </CommandGroup>
              <CommandSeparator />
            </>
          )}
          <div className="flex items-center justify-between bg-[var(--teal-a3)] p-2">
            <div>
              <Button
                className="size-7 rounded-full"
                disabled={currentPage === 1}
                onClick={() => setCurrentPage(1)}
                size="icon"
                variant="ghost"
              >
                <RxDoubleArrowLeft />
              </Button>
              <Button
                className="size-7 rounded-full"
                disabled={currentPage === 1}
                onClick={() => setCurrentPage(currentPage - 1)}
                size="icon"
                variant="ghost"
              >
                <RxCaretLeft />
              </Button>
            </div>
            <span>
              {currentPage} / {totalPages}
            </span>
            <div>
              <Button
                className="size-7 rounded-full"
                disabled={currentPage === totalPages}
                onClick={() => setCurrentPage(currentPage + 1)}
                size="icon"
                variant="ghost"
              >
                <RxCaretRight />
              </Button>
              <Button
                className="size-7 rounded-full"
                disabled={currentPage === totalPages}
                onClick={() => setCurrentPage(totalPages)}
                size="icon"
                variant="ghost"
              >
                <RxDoubleArrowRight />
              </Button>
            </div>
          </div>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
