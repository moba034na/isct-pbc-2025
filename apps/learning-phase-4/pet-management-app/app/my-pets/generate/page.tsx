"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Navbar } from "@/components/layout/navbar"
import { ArrowLeft, Sparkles } from "lucide-react"

interface Pet {
  id: string
  name: string
  category: string
  breed?: string
  imageUrl?: string
}

export default function GenerateChildPage() {
  const router = useRouter()
  const [user, setUser] = useState<any>(null)
  const [pets, setPets] = useState<Pet[]>([])
  const [parent1Id, setParent1Id] = useState<string>('')
  const [parent2Id, setParent2Id] = useState<string>('')
  const [generating, setGenerating] = useState(false)
  const [generatedImage, setGeneratedImage] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const userData = localStorage.getItem('user')
    if (!userData) {
      router.push('/login')
      return
    }
    const parsedUser = JSON.parse(userData)
    setUser(parsedUser)
    fetchPets(parsedUser.id)
  }, [router])

  const fetchPets = async (userId: string) => {
    try {
      const response = await fetch('/api/pets', {
        headers: {
          'x-user-id': userId,
        },
      })

      if (!response.ok) {
        throw new Error('Failed to fetch pets')
      }

      const data = await response.json()
      setPets(data.pets)
    } catch (error) {
      console.error('Fetch pets error:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleGenerate = async () => {
    if (!parent1Id || !parent2Id) {
      alert('Please select two pets')
      return
    }

    if (parent1Id === parent2Id) {
      alert('Please select different pets')
      return
    }

    const parent1 = pets.find((p) => p.id === parent1Id)
    const parent2 = pets.find((p) => p.id === parent2Id)

    setGenerating(true)
    setGeneratedImage(null)

    try {
      const response = await fetch('/api/pets/generate-child', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ parent1, parent2 }),
      })

      if (!response.ok) {
        throw new Error('Failed to generate image')
      }

      const data = await response.json()
      setGeneratedImage(data.imageUrl)
    } catch (error) {
      console.error('Generate error:', error)
      alert('Failed to generate image')
    } finally {
      setGenerating(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <div className="container mx-auto px-4 py-8">
          <div className="text-center">Loading...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <div className="container mx-auto px-4 py-8 max-w-2xl">
        <Link href="/my-pets">
          <Button variant="ghost" className="mb-4">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Pet List
          </Button>
        </Link>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Sparkles className="mr-2 h-5 w-5" />
              Generate Child Image
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <p className="text-sm text-gray-600">
              Select two pets and AI will generate an image of their child.
            </p>

            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Select Parent 1</label>
                <Select value={parent1Id} onValueChange={setParent1Id}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a pet..." />
                  </SelectTrigger>
                  <SelectContent>
                    {pets.map((pet) => (
                      <SelectItem key={pet.id} value={pet.id}>
                        {pet.name} ({pet.breed || pet.category})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Select Parent 2</label>
                <Select value={parent2Id} onValueChange={setParent2Id}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a pet..." />
                  </SelectTrigger>
                  <SelectContent>
                    {pets.map((pet) => (
                      <SelectItem key={pet.id} value={pet.id}>
                        {pet.name} ({pet.breed || pet.category})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <Button
              onClick={handleGenerate}
              disabled={!parent1Id || !parent2Id || generating}
              className="w-full"
            >
              {generating ? 'Generating...' : 'Generate Child Image'}
            </Button>

            {generatedImage && (
              <div className="space-y-2">
                <p className="text-sm font-medium">Generated Image:</p>
                <img
                  src={generatedImage}
                  alt="Generated child"
                  className="w-full rounded-lg shadow-md"
                />
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}