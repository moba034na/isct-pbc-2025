import { NextRequest, NextResponse } from 'next/server'
import { GoogleGenerativeAI } from '@google/generative-ai'

export async function POST(request: NextRequest) {
  try {
    const apiKey = process.env.GOOGLE_GEMINI_API_KEY

    if (!apiKey) {
      return NextResponse.json(
        { error: 'API key not configured' },
        { status: 500 }
      )
    }

    const formData = await request.formData()
    const file = formData.get('file') as File
    const category = formData.get('category') as string

    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      )
    }

    // Convert file to Base64
    const bytes = await file.arrayBuffer()
    const buffer = Buffer.from(bytes)
    const base64Image = buffer.toString('base64')

    // Call Gemini API
    const genAI = new GoogleGenerativeAI(apiKey)
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' })

    const prompt = category === 'Dog'
      ? 'Look at this dog image and identify the breed. Please respond with only the breed name in English. If there are multiple possibilities, answer with only the most likely one.'
      : category === 'Cat'
      ? 'Look at this cat image and identify the breed. Please respond with only the breed name in English. If there are multiple possibilities, answer with only the most likely one.'
      : 'Identify the species of this animal. Please respond with only the species name in English.'

    const result = await model.generateContent([
      {
        inlineData: {
          mimeType: file.type,
          data: base64Image,
        },
      },
      prompt,
    ])

    const breed = result.response.text().trim()

    return NextResponse.json({ breed })
  } catch (error) {
    console.error('Identify error:', error)
    return NextResponse.json(
      { error: 'Failed to identify breed' },
      { status: 500 }
    )
  }
}