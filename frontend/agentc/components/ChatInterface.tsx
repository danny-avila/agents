'use client';

import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";

interface Message {
  role: 'user' | 'ai';
  content: string;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');

  const sendMessage = async () => {
    console.log("sendMessage called"); // Debug log
    if (input.trim() === '') return;

    const newMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, newMessage]);

    try {
      console.log("Sending request to /api/chat"); // Debug log
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Received response:", data); // Debug log

      const aiMessage: Message = { role: 'ai', content: data.message };
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error("Error in sendMessage:", error);
      // Optionally add an error message to the chat
      setMessages(prev => [...prev, { role: 'ai', content: 'Sorry, an error occurred.' }]);
    }

    setInput('');
  };

  return (
    <div className="flex flex-col h-screen max-w-2xl mx-auto p-4">
      <ScrollArea className="flex-grow mb-4 p-4 border rounded">
        {messages.map((msg, index) => (
          <div key={index} className={`mb-2 ${msg.role === 'user' ? 'text-right' : 'text-left'}`}>
            <span className={`inline-block p-2 rounded ${msg.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}>
              {msg.content}
            </span>
          </div>
        ))}
      </ScrollArea>
      <div className="flex">
        <Input 
          value={input} 
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setInput(e.target.value)}
          onKeyPress={(e: React.KeyboardEvent<HTMLInputElement>) => e.key === 'Enter' && sendMessage()}
          placeholder="Type your message..."
          className="flex-grow mr-2"
        />
        <Button onClick={sendMessage}>Send</Button>
      </div>
    </div>
  );
}