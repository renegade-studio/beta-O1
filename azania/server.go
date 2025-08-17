package main

import (
	"context"
	"log"
	"net"

	"google.golang.org/grpc"
	pb "azania"
)

// server is used to implement contextengine.ContextEngineServer.
type server struct {
	pb.UnimplementedContextEngineServer
}

// ProcessText implements contextengine.ContextEngineServer
func (s *server) ProcessText(ctx context.Context, in *pb.ProcessTextRequest) (*pb.ProcessTextResponse, error) {
	log.Printf("Received: %v", in.GetText())
	return &pb.ProcessTextResponse{ProcessedText: "Echo: " + in.GetText()}, nil
}

func main() {
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	s := grpc.NewServer()
	pb.RegisterContextEngineServer(s, &server{})
	log.Printf("server listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
