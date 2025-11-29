import { NextRequest, NextResponse } from 'next/server'
import { GoogleGenerativeAI } from '@google/generative-ai'

export async function POST(request: NextRequest) {
  try {
    const hfApiKey = process.env.HUGGINGFACE_API_KEY
    const geminiApiKey = process.env.GOOGLE_GEMINI_API_KEY

    if (!hfApiKey || !geminiApiKey) {
      return NextResponse.json(
        { error: 'API key not configured' },
        { status: 500 }
      )
    }

    const { parent1, parent2 } = await request.json()

    if (!parent1 || !parent2) {
      return NextResponse.json(
        { error: 'Two parents are required' },
        { status: 400 }
      )
    }

    // Check if categories match
    const sameCategory = parent1.category === parent2.category

    // Extract features from parent images with Gemini API
    let prompt = sameCategory
      ? `A cute baby ${parent1.category.toLowerCase()}`
      : `A creature that is a mix of a ${parent1.category.toLowerCase()} and a ${parent2.category.toLowerCase()}`

    // If parents have images, analyze features with Gemini
    if (parent1.imageUrl && parent2.imageUrl) {
      try {
        const genAI = new GoogleGenerativeAI(geminiApiKey)
        const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' })

        // Fetch images
        const [img1Response, img2Response] = await Promise.all([
          fetch(parent1.imageUrl),
          fetch(parent2.imageUrl),
        ])

        const [img1Buffer, img2Buffer] = await Promise.all([
          img1Response.arrayBuffer(),
          img2Response.arrayBuffer(),
        ])

        const img1Base64 = Buffer.from(img1Buffer).toString('base64')
        const img2Base64 = Buffer.from(img2Buffer).toString('base64')

        // Analyze parent features with Gemini
        const analysisPrompt = `Look at these two pet images and describe their visual characteristics (fur color, patterns, eye color, body type, etc.) concisely in English.

Image 1: ${parent1.name} (${parent1.breed || parent1.category})
Image 2: ${parent2.name} (${parent2.breed || parent2.category})

Please respond in the following format:
Parent 1: [fur color], [pattern characteristics], [other features]
Parent 2: [fur color], [pattern characteristics], [other features]
Child (mix): [imagined appearance of child combining features of both]`

        const result = await model.generateContent([
          {
            inlineData: {
              mimeType: 'image/jpeg',
              data: img1Base64,
            },
          },
          {
            inlineData: {
              mimeType: 'image/jpeg',
              data: img2Base64,
            },
          },
          analysisPrompt,
        ])

        const analysis = result.response.text()
        console.log('Gemini analysis:', analysis)

        // Extract Child part from analysis
        const childMatch = analysis.match(/Child.*?:(.*?)(?:\n|$)/i)
        if (childMatch) {
          const childDescription = childMatch[1].trim()
          // Add breed information
          const breed1 = parent1.breed || parent1.category
          const breed2 = parent2.breed || parent2.category
          const breedInfo = `mix of ${breed1} and ${breed2}`

          if (sameCategory) {
            prompt = `A cute baby ${parent1.category.toLowerCase()} (${breedInfo}), ${childDescription}, adorable, fluffy, high quality, professional photo, cute face, detailed fur texture`
          } else {
            prompt = `A creature that is a mix of a ${parent1.category.toLowerCase()} and a ${parent2.category.toLowerCase()} (${breedInfo}), ${childDescription}, adorable, high quality, professional photo, detailed fur texture`
          }
        }
      } catch (error) {
        console.error('Gemini analysis error:', error)
        // Fallback on error
        if (sameCategory) {
          prompt = `A cute baby ${parent1.category.toLowerCase()} that is a mix between a ${parent1.breed || parent1.category} and a ${parent2.breed || parent2.category}, adorable, fluffy, high quality, professional photo`
        } else {
          prompt = `A creature that is a mix of a ${parent1.category.toLowerCase()} and a ${parent2.category.toLowerCase()}, combining features of a ${parent1.breed || parent1.category} and a ${parent2.breed || parent2.category}, adorable, high quality, professional photo`
        }
      }
    } else {
      // Breed name based if no images
      if (sameCategory) {
        prompt = `A cute baby ${parent1.category.toLowerCase()} that is a mix between a ${parent1.breed || parent1.category} and a ${parent2.breed || parent2.category}, adorable, fluffy, high quality, professional photo`
      } else {
        prompt = `A creature that is a mix of a ${parent1.category.toLowerCase()} and a ${parent2.category.toLowerCase()}, combining features of a ${parent1.breed || parent1.category} and a ${parent2.breed || parent2.category}, adorable, high quality, professional photo`
      }
    }

    console.log('Final prompt:', prompt)

    // Call Hugging Face Inference API (SDXL Base 1.0)
    const response = await fetch(
      'https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0',
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${hfApiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          inputs: prompt,
          parameters: {
            negative_prompt: 'ugly, deformed, low quality, blurry, distorted',
            num_inference_steps: 30,
            width: 1024,
            height: 1024,
          },
        }),
      }
    )

    if (!response.ok) {
      const errorText = await response.text()
      console.error('Hugging Face API error:', response.status, errorText)
      throw new Error(`Failed to generate image: ${response.status} - ${errorText}`)
    }

    // Get image data
    const imageBuffer = await response.arrayBuffer()
    const base64Image = Buffer.from(imageBuffer).toString('base64')
    const imageUrl = `data:image/jpeg;base64,${base64Image}`

    return NextResponse.json({ imageUrl })
  } catch (error) {
    console.error('Generate error:', error)
    return NextResponse.json(
      { error: 'Failed to generate child image' },
      { status: 500 }
    )
  }
}