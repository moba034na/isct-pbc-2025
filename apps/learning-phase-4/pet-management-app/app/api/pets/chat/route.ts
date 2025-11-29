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

    const { message, petInfo } = await request.json()

    if (!message) {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      )
    }

    // Call Gemini API
    const genAI = new GoogleGenerativeAI(apiKey)
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' })

    const systemPrompt = `You are a pet health advisor. Please kindly answer the owner's questions based on the following pet information.

Pet Information:
- Name: ${petInfo.name}
- Category: ${petInfo.category}
- Breed: ${petInfo.breed || 'Unknown'}
- Gender: ${petInfo.gender || 'Unknown'}
- Age: ${petInfo.age || 'Unknown'}

Important Notes:
- Provide only general advice
- If symptoms are urgent, always encourage consulting a veterinarian
- Do not provide diagnoses or prescriptions
- Explain in gentle, easy-to-understand language`

    const result = await model.generateContent([
      systemPrompt,
      `Question: ${message}`,
    ])

    const response = result.response.text()

    return NextResponse.json({ response })
  } catch (error) {
    console.error('Chat error:', error)
    return NextResponse.json(
      { error: 'Failed to get response' },
      { status: 500 }
    )
  }
}