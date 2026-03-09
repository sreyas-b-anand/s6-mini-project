"use client"

import { SquareArrowOutUpRight } from 'lucide-react'
import Link from 'next/link'
import React from 'react'

const Navbar = () => {
  return (
    <div className='w-full py-4 flex justify-between items-center shadow-md px-6 rounded-md'>
      <p className='font-semibold text-2xl'>RevDet</p>

      <div className='bg-foreground text-background py-3 px-4 rounded-md flex items-center justify-center gap-2 text-sm font-medium hover:opacity-95 '>
        <Link href={'/bert'} >Try BERT Model</Link>
        <SquareArrowOutUpRight size={18} />
      </div>
    </div>
  )
}

export default Navbar
