'use client'
import { Button } from "@/components/ui/button"

import { Table } from '@tanstack/react-table'

import { RxCheck, RxCaretLeft, RxCaretRight, RxMagnifyingGlass } from "react-icons/rx";

import { useState, useMemo } from "react"

import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"

import { Separator } from "@/components/ui/separator"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface DataTableFacetedFilterProps<TData> {
  table: Table<TData>
  columnName: string
}

export function DataTableFacetedFilter<TData>({
  table,
  columnName
}: DataTableFacetedFilterProps<TData>) {
  const column = table.getColumn(columnName)
  if (!column) return null

  // Create a map object to store the frecency of each option
  let optionFrequency: Map<string, number> = new Map()
  if (columnName === 'authors') {
    // Iterate over the original map object having the authors array as keys
    column?.getFacetedUniqueValues().forEach((frequency, authors) => {
      // Iterate over the authors array
      authors.forEach((author: string) => {
        // If the author is already in the map object, increment its frecency
        if (optionFrequency.has(author)) {
          optionFrequency.set(author, optionFrequency.get(author)! + frequency)
        } else {
          // Otherwise, set the frecency to the frequency
          optionFrequency.set(author, frequency)
        }
      })
    })
  } else {
    optionFrequency = column?.getFacetedUniqueValues()
  }

  const options = useMemo(() => Array.from(optionFrequency.keys()), [optionFrequency]);
  const selectedValues = new Set<string>(column?.getFilterValue() as string[])

  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1)
  const itemsPerPage = 20


  const filteredOptions = useMemo(() => {
    return options.sort().filter(option => option.toLowerCase().includes(searchTerm.toLowerCase()));
  }, [options, searchTerm]);

  const paginatedOptions = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return filteredOptions.slice(startIndex, startIndex + itemsPerPage);
  }, [filteredOptions, currentPage, itemsPerPage]);

  const totalPages = Math.ceil(filteredOptions.length / itemsPerPage)

  const handleSearchTermChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.target.value);
    setCurrentPage(1);
  }

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm" className="h-8 w-fit border-dashed rounded-sm bg-[var(--teal-4)]">
          {String(column.columnDef.header)}
          {selectedValues?.size > 0 && (
            <>
              <Separator orientation="vertical" className="mx-2 h-4" />
              <Badge
                variant="default"
                className="rounded-sm px-1 font-normal bg-[var(--teal-9)] sm:hidden"
              >
                {selectedValues.size}
              </Badge>
              <div className="hidden space-x-1 sm:flex">
                {selectedValues.size > 2 ? (
                  <Badge
                    variant="default"
                    className="rounded-sm px-1 bg-[var(--teal-9)] font-normal"
                  >
                    {selectedValues.size} selected
                  </Badge>
                ) : (
                  Array.from(options)
                    .sort()
                    .filter((option) => selectedValues.has(option))
                    .map((option) => (
                      <Badge
                        variant="default"
                        key={option}
                        className="rounded-sm px-1 bg-[var(--teal-9)] font-normal"
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
      <PopoverContent className="w-[250px] p-0" align="start">
        <Command>
          <div className="flex items-center border-b px-3">
            <RxMagnifyingGlass className="mr-2 h-4 w-4 shrink-0 opacity-50" />
            <input
              placeholder="Search..."
              value={searchTerm}
              onChange={handleSearchTermChange}
              className="flex h-11 w-full rounded-md bg-transparent py-3 text-sm outline-none placeholder:text-muted-foreground font-medium disabled:cursor-not-allowed disabled:opacity-50"
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
                        "mr-2 flex h-4 w-4 items-center justify-center rounded-sm border border-primary",
                        isSelected
                          ? "bg-primary text-primary-foreground"
                          : "opacity-50 [&_svg]:invisible"
                      )}
                    >
                      <RxCheck className={cn("h-4 w-4")} />
                    </div>
                    <span>{option}</span>
                    <span className="ml-auto flex h-4 w-4 items-center justify-center font-mono text-xs">
                      {optionFrequency.get(option)}
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
                  onSelect={() => column?.setFilterValue(undefined)}
                  className="justify-center text-center"
                >
                  Clear filters
                </CommandItem>
              </CommandGroup>
            </>
          )}
          <div className="flex justify-between items-center p-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage(currentPage - 1)}
              disabled={currentPage === 1}
            >
              <RxCaretLeft/>
            </Button>
            <span>{currentPage} / {totalPages}</span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage(currentPage + 1)}
              disabled={currentPage === totalPages}
            >
              <RxCaretRight/>
            </Button>
          </div>
        </Command>
      </PopoverContent>
    </Popover>
  )
}