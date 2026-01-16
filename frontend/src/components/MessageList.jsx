import Message from './Message'

function MessageList({ messages }) {
  return (
    <div className="space-y-4 max-w-4xl mx-auto">
      {messages.map((message, index) => (
        <Message key={index} message={message} />
      ))}
    </div>
  )
}

export default MessageList
